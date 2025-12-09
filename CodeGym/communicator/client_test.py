import requests

# 容器A的接口地址（因在同一网络，可直接用容器名访问）
CONTAINER_A_URL = "http://code-gym:8000/execute-command"
# 鉴权Key（需与容器A的AUTH_API_KEY一致）
API_KEY = "container-a-secure-key-2025"

def call_container_a(command: str, timeout: int = 30):
    """
    容器B调用容器A的接口，执行命令
    :param command: 要在容器A内执行的命令
    :param timeout: 命令超时时间
    :return: 容器A返回的结果
    """
    # 请求头（携带鉴权Key）
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    # 请求体（传递命令和超时时间）
    data = {
        "command": command,
        "timeout": timeout
    }
    
    try:
        # 发送POST请求
        response = requests.post(
            CONTAINER_A_URL,
            json=data,
            headers=headers,
            timeout=35  # 整体请求超时（略大于命令超时）
        )
        # 返回JSON结果
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "code": 500,
            "msg": f"调用容器A失败：{str(e)}",
            "data": None
        }

# 测试调用：在容器A内执行ls ./
if __name__ == "__main__":
    result = call_container_a("ls ./")
    print("容器A执行结果：")
    print(f"状态码：{result['code']}")
    print(f"提示：{result['msg']}")
    if result["data"]:
        print(f"退出码：{result['data']['exit_code']}")
        print(f"标准输出：\n{result['data']['stdout']}")
        print(f"标准错误：\n{result['data']['stderr']}")
