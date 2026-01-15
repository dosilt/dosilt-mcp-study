# server.py
from fastmcp import FastMCP
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("PersonalCalculator")


class CalculatorState:
    def __init__(self):
        self.user_name: Optional[str] = None
        self.history: list[dict] = []
        self.total: float = 0.0

    def reset(self):
        self.history = []
        self.total = 0.0


state = CalculatorState()


@mcp.tool()
def set_user_name(name: str) -> str:
    """Set your name for personalized calculator experience"""
    state.user_name = name
    logger.info(f"User name set to: {name}")
    return f"안녕하세요, {name}님! 계산기를 시작합니다."


@mcp.tool()
def get_user_name() -> str:
    """Get current user name"""
    if state.user_name:
        return f"현재 사용자: {state.user_name}"
    else:
        return "아직 이름이 설정되지 않았습니다."


@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers"""
    result = a + b
    state.total += result
    state.history.append({"operation": "add", "values": [a, b], "result": result})

    greeting = f"{state.user_name}님, " if state.user_name else ""
    return f"{greeting}{a} + {b} = {result}"


@mcp.tool()
def subtract(a: float, b: float) -> str:
    """Subtract b from a"""
    result = a - b
    state.total += result
    state.history.append({"operation": "subtract", "values": [a, b], "result": result})

    greeting = f"{state.user_name}님, " if state.user_name else ""
    return f"{greeting}{a} - {b} = {result}"


@mcp.tool()
def multiply(a: float, b: float) -> str:
    """Multiply two numbers"""
    result = a * b
    state.total += result
    state.history.append({"operation": "multiply", "values": [a, b], "result": result})

    greeting = f"{state.user_name}님, " if state.user_name else ""
    return f"{greeting}{a} × {b} = {result}"


@mcp.tool()
def divide(a: float, b: float) -> str:
    """Divide a by b"""
    if b == 0:
        return "❌ 0으로 나눌 수 없습니다!"

    result = a / b
    state.total += result
    state.history.append({"operation": "divide", "values": [a, b], "result": result})

    greeting = f"{state.user_name}님, " if state.user_name else ""
    return f"{greeting}{a} ÷ {b} = {result}"


@mcp.tool()
def get_history() -> str:
    """Get calculation history"""
    if not state.history:
        return "계산 기록이 없습니다."

    result = []
    if state.user_name:
        result.append(f"📊 {state.user_name}님의 계산 기록:\n")
    else:
        result.append("📊 계산 기록:\n")

    for i, record in enumerate(state.history, 1):
        op = record["operation"]
        vals = record["values"]
        res = record["result"]
        result.append(f"{i}. {op}: {vals[0]} → {vals[1]} = {res}")

    return "\n".join(result)


@mcp.tool()
def get_total() -> str:
    """Get total sum of all calculation results"""
    greeting = f"{state.user_name}님, " if state.user_name else ""
    return f"{greeting}모든 계산 결과의 합: {state.total}"


@mcp.tool()
def get_stats() -> str:
    """Get calculator statistics"""
    if not state.history:
        return "통계 데이터가 없습니다."

    op_counts = {}
    for record in state.history:
        op = record["operation"]
        op_counts[op] = op_counts.get(op, 0) + 1

    result = []
    if state.user_name:
        result.append(f"📈 {state.user_name}님의 통계:")
    else:
        result.append("📈 계산기 통계:")

    result.append(f"- 총 계산 횟수: {len(state.history)}")
    result.append(f"- 누적 합계: {state.total}")
    result.append("- 연산별 사용 횟수:")
    for op, count in op_counts.items():
        result.append(f"  • {op}: {count}회")

    return "\n".join(result)


@mcp.tool()
def reset_calculator() -> str:
    """Reset calculator (clear history and total, keep user name)"""
    state.reset()
    greeting = f"{state.user_name}님, " if state.user_name else ""
    return f"{greeting}계산기가 초기화되었습니다."


@mcp.tool()
def reset_all() -> str:
    """Reset everything including user name"""
    old_name = state.user_name
    state.user_name = None
    state.reset()

    if old_name:
        return f"안녕히 가세요, {old_name}님! 모든 데이터가 초기화되었습니다."
    else:
        return "모든 데이터가 초기화되었습니다."


if __name__ == "__main__":
    logger.info("Personal Calculator MCP Server starting...")
    mcp.run(transport="stdio")
