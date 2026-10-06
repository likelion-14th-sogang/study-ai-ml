import sys

if sys.platform == "win32" and "resource" not in sys.modules:
    import types

    resource_stub = types.ModuleType("resource")
    resource_stub.RLIMIT_AS = 9  # 값 자체는 안 쓰임 (Windows에서 관련 테스트는 skip됨)

    class error(Exception):
        pass

    resource_stub.error = error

    def getrlimit(which):
        return (-1, -1)

    def setrlimit(which, limits):
        pass

    resource_stub.getrlimit = getrlimit
    resource_stub.setrlimit = setrlimit

    sys.modules["resource"] = resource_stub