from proxy_app.routers.auth_api import router as auth_router
from proxy_app.routers.admin_api import router as admin_router
from proxy_app.routers.user_api import router as user_router
from proxy_app.routers.ui import router as ui_router

__all__ = ["auth_router", "user_router", "admin_router", "ui_router"]
