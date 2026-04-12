# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""API router and state init for streaming video realtime (WebSocket)."""

from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, WebSocket

from vllm.entrypoints.openai.video_realtime.connection import (
    RealtimeVideoConnection,
)
from vllm.entrypoints.openai.video_realtime.serving import (
    OpenAIServingRealtimeVideo,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object

router = APIRouter()


@router.websocket("/v1/realtime_video")
async def realtime_video_endpoint(websocket: WebSocket):
    """WebSocket endpoint for realtime video understanding.

    Protocol:
      1. Client connects to ws://host/v1/realtime_video
      2. Server sends session.created
      3. Client sends frames: input_video_buffer.append with base64-encoded
         frames (e.g. JPEG), then input_video_buffer.commit to process buffer
      4. Server sends completion.delta then completion.done
      5. Repeat from step 3; use input_video_buffer.commit with final=True
         when done
      6. Client can send generation.trigger to set/trigger a new prompt for
         the next generation cycle

    Text-only: send generation.trigger with prompt, then
    input_video_buffer.commit with no frames in buffer.
    Video: append frames then commit. Empty commit = text-only turn.
    """
    app = websocket.app
    serving = getattr(app.state, "openai_serving_realtime_video", None)
    if serving is None:
        await websocket.accept()
        await websocket.send_text(
            '{"type":"error","error":"Realtime video not enabled","code":"not_available"}'
        )
        await websocket.close()
        return
    connection = RealtimeVideoConnection(websocket, serving)
    await connection.handle_connection()


def attach_router(app: FastAPI):
    """Attach the realtime_video router to the FastAPI app."""
    app.include_router(router)
    logger.info("Realtime video API router attached")


def init_realtime_video_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    """Initialize openai_serving_realtime_video when generate task is supported."""
    state.openai_serving_realtime_video = (
        OpenAIServingRealtimeVideo(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            log_error_stack=args.log_error_stack,
        )
        if "realtime_video" in supported_tasks
        else None
    )