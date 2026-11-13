"""Only safe, user-facing messages cross the adapter boundary."""


class AppError(Exception):
    pass


class AccessDenied(AppError):
    pass


class InvalidRequest(AppError):
    pass


class UnsafeQuery(AppError):
    pass


class QueryFailed(AppError):
    """The model may correct the query, within its SQL attempt limit."""


class ServiceUnavailable(AppError):
    pass


class BudgetExceeded(AppError):
    pass
