# client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    def __init__(self, model_name="Qwen", server_script="server.py"):
        self.model = self.select_model(model_name)

        self.server_params = StdioServerParameters(
            command="python",
            args=[server_script],
        )

        self.agent = None
        self.stdio_ctx = None
        self.session_ctx = None
        self.is_running = False

    def select_model(self, model_name):
        if "gpt" in model_name:
            model = ChatOpenAI(
                model="gpt-4.1-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        else:
            model = ChatOpenAI(
                model="Qwen/Qwen3-32B",  # vLLM에서 로드한 모델명
                base_url=os.getenv("CUSTOM_LLM_URL"),  # vLLM 서버 주소
                api_key="EMPTY",  # vLLM은 API key 불필요
                temperature=0.7,
            )

        return model

    async def start(self):
        """MCP 세션 시작"""
        if self.is_running:
            print("⚠️ 이미 실행 중입니다!")
            return

        print("🚀 MCP 서버 연결 중...")
        self.stdio_ctx = stdio_client(self.server_params)
        read, write = await self.stdio_ctx.__aenter__()

        self.session_ctx = ClientSession(read, write)
        session = await self.session_ctx.__aenter__()

        await session.initialize()
        tools = await load_mcp_tools(session)
        self.agent = create_react_agent(self.model, tools)

        self.is_running = True
        print("✅ MCP 세션이 시작되었습니다!\n")

    async def ask(self, message: str, show_message=True) -> str:
        """에이전트에게 질문"""
        if not self.is_running:
            return "❌ 먼저 start()를 실행하세요!"

        if show_message:
            print(f"💬 질문: {message}")

        response = await self.agent.ainvoke({"messages": message})
        result = response["messages"][-1].content

        if show_message:
            print(f"🤖 답변: {result}\n")

        return result

    async def stop(self):
        """세션 종료"""
        if not self.is_running:
            return

        if self.session_ctx:
            await self.session_ctx.__aexit__(None, None, None)
        if self.stdio_ctx:
            await self.stdio_ctx.__aexit__(None, None, None)

        self.is_running = False
        print("✅ MCP 세션이 종료되었습니다!")


async def main():
    """메인 시나리오"""
    client = MCPClient()

    try:
        # 세션 시작
        await client.start()

        # 시나리오 1: 이름 설정
        print("=" * 50)
        print("📝 시나리오 1: 이름 설정")
        print("=" * 50)
        await client.ask("내 이름은 철수야")

        # 시나리오 2: 계산하기
        print("=" * 50)
        print("🧮 시나리오 2: 계산하기")
        print("=" * 50)
        await client.ask("5 + 3을 계산해줘")
        await client.ask("10 × 2를 계산해줘")
        await client.ask("20 - 5를 계산해줘")

        # 시나리오 3: 기록 확인
        print("=" * 50)
        print("📊 시나리오 3: 기록 확인")
        print("=" * 50)
        await client.ask("내 계산 기록을 보여줘")

        # 시나리오 4: 통계 확인
        print("=" * 50)
        print("📈 시나리오 4: 통계 확인")
        print("=" * 50)
        await client.ask("통계를 보여줘")

        # 시나리오 5: 누적 합계
        print("=" * 50)
        print("💰 시나리오 5: 누적 합계")
        print("=" * 50)
        await client.ask("지금까지 계산한 결과의 총합은?")

        # 시나리오 6: 이름 확인
        print("=" * 50)
        print("👤 시나리오 6: 이름 확인")
        print("=" * 50)
        await client.ask("내 이름이 뭐야?")

    finally:
        # 세션 종료
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
