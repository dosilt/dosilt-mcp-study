# client_cli.py
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
        print("✅ MCP 세션이 시작되었습니다!")

    async def ask(self, message: str) -> str:
        """에이전트에게 질문"""
        if not self.is_running:
            return "❌ 먼저 start()를 실행하세요!"

        response = await self.agent.ainvoke({"messages": message})
        return response["messages"][-1].content

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


async def interactive_mode():
    """대화형 CLI 모드"""
    client = MCPClient()

    print("=" * 60)
    print("🧮 Personal Calculator MCP Client")
    print("=" * 60)
    print()
    print("명령어:")
    print("  - 일반 질문: 자유롭게 입력하세요")
    print("  - 'exit' 또는 'quit': 종료")
    print("  - 'help': 도움말")
    print()

    try:
        await client.start()
        print()

        while True:
            try:
                user_input = input("💬 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\n👋 계산기를 종료합니다...")
                    break

                if user_input.lower() == "help":
                    print("\n📖 사용 가능한 기능:")
                    print("  - 이름 설정: '내 이름은 철수야'")
                    print("  - 계산: '5 + 3', '10 × 2', '20 - 5', '100 ÷ 4'")
                    print("  - 기록: '기록 보여줘', '내 계산 기록'")
                    print("  - 통계: '통계 보여줘'")
                    print("  - 총합: '누적 합계는?', '총합은?'")
                    print("  - 초기화: '계산기 초기화', '모두 초기화'")
                    print()
                    continue

                print("🤔 처리 중...", end="\r")
                result = await client.ask(user_input)
                print(" " * 20, end="\r")  # 처리 중 메시지 지우기
                print(f"🤖 Bot: {result}\n")

            except KeyboardInterrupt:
                print("\n\n⚠️ Ctrl+C 감지. 종료하려면 'exit'를 입력하세요.\n")
                continue
            except Exception as e:
                print(f"\n❌ 오류: {e}\n")
                continue

    finally:
        await client.stop()


async def main():
    """메인 함수"""
    await interactive_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
