import asyncio
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from mcp.client.sse import sse_client
import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv("../.env")


class MCPClient:
    def __init__(self):
        self.model = self.select_model(os.getenv("MODEL_NAME", "gpt-4"))
        self.agent = None
        self.sse_ctx = None
        self.session_ctx = None
        self.is_running = False
        self.thread_id = None

    def select_model(self, model_name):
        """모델 선택"""
        if "gpt" in model_name or "o1" in model_name:
            model = ChatOpenAI(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.7,
                streaming=True,
                model_kwargs={
                    "parallel_tool_calls": False  # 순차 실행
                },
            )
        else:
            model = ChatOpenAI(
                model=model_name,
                base_url=os.getenv("CUSTOM_LLM_URL"),
                api_key="EMPTY",
                temperature=0.7,
                streaming=True,
            )
        return model

    async def start(self, reset_server=True):
        """MCP 세션 시작"""
        print("🔌 서버에 연결 중...")

        # SSE 연결
        self.sse_ctx = sse_client(url="http://localhost:8234/sse")
        read, write = await self.sse_ctx.__aenter__()

        # MCP 세션
        self.session_ctx = ClientSession(read, write)
        session = await self.session_ctx.__aenter__()

        await session.initialize()

        # Tool 로드
        tools = await load_mcp_tools(session)
        print(f"🔧 {len(tools)}개 도구 로드됨")

        # Agent 생성
        self.agent = create_react_agent(self.model, tools)

        # Thread ID 생성
        self.thread_id = str(uuid.uuid4())

        self.is_running = True
        print(f"✅ MCP 세션 시작! (Thread: {self.thread_id[:8]}...)\n")

        # 서버 초기화
        if reset_server:
            await self._reset_server()

    async def _reset_server(self):
        """서버 데이터 초기화"""
        print("🔄 서버 데이터 초기화 중...")
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            response = await self.agent.ainvoke(
                {"messages": [("user", "clear_all_data를 실행해주세요")]}, config=config
            )
            result = response["messages"][-1].content
            print(result)
        except Exception as e:
            print(f"⚠️ 초기화 실패: {e}")
        print()

    async def ask_with_streaming(self, message: str) -> str:
        """✨✨ 실시간 스트리밍 + Tool 강제 사용"""
        if not self.is_running:
            return "❌ 먼저 start()를 실행하세요!"

        print(f"\n{'=' * 70}")
        print(f"💬 질문: {message}")
        print(f"{'=' * 70}\n")

        # ✅ Tool 사용 강제 프롬프트
        enhanced_message = f"""CRITICAL RULES:
1. You MUST use the available tools for ALL calculations
2. Do NOT calculate anything in your head
3. Do NOT write numbers as results without calling tools
4. Before each tool call, explain your reasoning
5. After each tool result, explain what you learned

Available calculation tools:
- add(a, b): addition
- subtract(a, b): subtraction  
- multiply(a, b): multiplication
- divide(a, b): division
- percentage(value, percent): calculate percentage
- increase_by_percent(value, percent): increase by %
- decrease_by_percent(value, percent): decrease by %
- calculate_average(numbers): average of list
- calculate_sum(numbers): sum of list
- find_max(numbers): maximum value
- find_min(numbers): minimum value
- compare_numbers(a, b): compare two numbers
- is_greater_than(value, threshold): check if greater
- is_less_than(value, threshold): check if less

Task: {message}

Remember: USE TOOLS FOR EVERY CALCULATION! Explain your reasoning before each tool call."""

        config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 100}

        thinking_num = 0
        action_num = 0
        current_thinking = ""

        print("🌊 Streaming started...\n")

        async for event in self.agent.astream_events(
            {"messages": [("user", enhanced_message)]}, config=config, version="v2"
        ):
            kind = event["event"]

            # 🧠 LLM 시작
            if kind == "on_chat_model_start":
                thinking_num += 1
                current_thinking = ""
                print(f"{'─' * 70}")
                print(f"💭 Thought #{thinking_num}:")
                print("   ", end="", flush=True)

            # 🌊 LLM 스트리밍
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]

                if hasattr(chunk, "content") and chunk.content:
                    print(chunk.content, end="", flush=True)
                    current_thinking += chunk.content

            # ✅ LLM 종료
            elif kind == "on_chat_model_end":
                print()  # 개행

                if not current_thinking.strip():
                    print("   (No reasoning - function calling mode)")

                output = event["data"].get("output")
                if output and hasattr(output, "tool_calls") and output.tool_calls:
                    print()
                    for tc in output.tool_calls:
                        action_num += 1
                        print(f"🔧 Action #{action_num}: {tc['name']}")
                        args_str = ", ".join([
                            f"{k}={v}" for k, v in tc["args"].items()
                        ])
                        print(f"   Args: {args_str}")

            # ✅ Tool 종료
            elif kind == "on_tool_end":
                tool_output = event["data"].get("output")
                if isinstance(tool_output, list):
                    tool_output = tool_output[0].get("text", str(tool_output))

                print(f"\n📊 Observation:")
                print(f"   {tool_output}\n")

        print(f"{'=' * 70}")
        print(f"✅ 완료! (Thoughts: {thinking_num}, Actions: {action_num})")

        if action_num == 0:
            print(f"\n⚠️  경고: Tool이 하나도 사용되지 않았습니다!")
            print(f"💡  LLM이 직접 계산했을 가능성이 높습니다.")

        print(f"{'=' * 70}\n")

    async def stop(self):
        """세션 종료"""
        if self.session_ctx:
            await self.session_ctx.__aexit__(None, None, None)

        if self.sse_ctx:
            await self.sse_ctx.__aexit__(None, None, None)

        self.is_running = False
        print("👋 MCP 세션이 종료되었습니다!")


async def main():
    """복잡한 계산으로 Tool 사용 강제"""
    client = MCPClient()

    try:
        await client.start(reset_server=True)

        # 테스트 1: 다단계 계산
        print("\n" + "🟢" * 35)
        print("🧠 테스트 1: 다단계 계산 (Tool 강제)")
        print("🟢" * 35)
        await client.ask_with_streaming(
            "다음을 순서대로 계산해줘:\n"
            "1. 123 + 456\n"
            "2. 결과 - 78\n"
            "3. 결과 ÷ 12\n"
            "4. 결과 × 25"
        )

        # 테스트 2: 백분율 계산
        print("\n" + "🔵" * 35)
        print("📊 테스트 2: 백분율 계산")
        print("🔵" * 35)
        await client.ask_with_streaming(
            "150을 23% 증가시킨 다음, 그 결과에서 18을 빼줘"
        )

        # 테스트 3: 여러 값 비교
        print("\n" + "🟡" * 35)
        print("⚖️ 테스트 3: 여러 값 비교")
        print("🟡" * 35)
        await client.ask_with_streaming(
            "88 × 12, 1500 ÷ 3, 100 × 10을 각각 계산하고,\n"
            "그 중 최댓값, 최솟값, 평균을 구해줘"
        )

        # 테스트 4: 기록 분석
        print("\n" + "🟣" * 35)
        print("📈 테스트 4: 기록 분석")
        print("🟣" * 35)
        await client.ask_with_streaming(
            "내 계산 기록을 보여주고, 통계를 분석하고, 총합을 계산해줘.\n"
            "총합이 5000보다 크면 '상위권', 3000~5000이면 '중위권', 아니면 '하위권'으로 분류해줘"
        )

    finally:
        await client.stop()


async def simple_test():
    """간단한 테스트 - Tool 사용 확인"""
    client = MCPClient()

    try:
        await client.start(reset_server=True)

        print("\n" + "🔵" * 35)
        print("🧪 간단한 테스트: Tool 사용 확인")
        print("🔵" * 35)
        await client.ask_with_streaming("10 + 20을 계산해줘")

        print("\n" + "🟢" * 35)
        print("🧪 기록 확인")
        print("🟢" * 35)
        await client.ask_with_streaming("내 계산 기록을 보여줘")

    finally:
        await client.stop()


if __name__ == "__main__":
    # 간단한 테스트 먼저
    # asyncio.run(simple_test())

    # 또는 전체 테스트
    asyncio.run(main())
