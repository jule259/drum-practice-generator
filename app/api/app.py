"""FastAPI application factory."""

from __future__ import annotations

import traceback

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import APP_NAME, __version__, config
from ..errors import DrumPracticeError
from ..logging_setup import get_logger, setup
from .routes import router

logger = get_logger("app")


def create_app() -> FastAPI:
    """Build the ASGI app: API routes plus the static frontend."""
    config.ensure_directories()
    setup()

    app = FastAPI(
        title=APP_NAME,
        version=__version__,
        description="本地 AI 电子鼓练习曲生成器",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # ---- Expected, user-actionable failures -------------------------
    @app.exception_handler(DrumPracticeError)
    async def _handle_domain_error(_request: Request, exc: DrumPracticeError):
        logger.warning("业务错误 [%s]：%s", exc.code, exc.message)
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})

    # ---- Anything else: still return JSON, never an HTML traceback ---
    @app.exception_handler(Exception)
    async def _handle_unexpected(_request: Request, exc: Exception):
        detail = traceback.format_exc()
        logger.error("未处理异常：%s\n%s", exc, detail)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "unexpected",
                    "message": f"服务器内部错误：{type(exc).__name__}: {exc}",
                    "suggestions": [
                        "重试一次。",
                        "查看 logs\\drum-practice.log 的完整堆栈。",
                        "运行 diagnose.bat 检查环境。",
                    ],
                    "detail": detail[-4000:],
                }
            },
        )

    app.include_router(router)

    # ---- Static frontend --------------------------------------------
    frontend = config.FRONTEND_DIR
    if frontend.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(frontend)),
            name="static",
        )

        @app.get("/", include_in_schema=False)
        async def index():
            index_file = frontend / "index.html"
            if not index_file.is_file():
                return JSONResponse(
                    status_code=500,
                    content={"error": {"code": "frontend_missing", "message": "frontend/index.html 缺失"}},
                )
            # No caching: the UI is under active development.
            return FileResponse(
                index_file,
                media_type="text/html",
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            icon = frontend / "favicon.ico"
            if icon.is_file():
                return FileResponse(icon)
            return JSONResponse(status_code=404, content={})
    else:
        logger.warning("未找到 frontend 目录：%s", frontend)

    return app
