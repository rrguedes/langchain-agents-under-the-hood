import ollama
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


def get_product_price(product: str) -> float:
    """Look up thew price of a product in the catalog"""
    print(f">>> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }
    return prices.get(product, 0)


def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold
    """
    print(
        f">>> Executing apply_discount(price='{price}', discount_tier='{discount_tier}')"
    )
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


llm_tools = [
    get_product_price,
    apply_discount,
]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=llm_tools, messages=messages)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received a price from get_product_price. "
                "PASS the exact price returned by get_product_price - DO NOT pass a made-up number.\n"
                "3. Never calculate discounts yourself using math."
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discounbt tier, "
                "ask them which tier to use. Do not assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {iteration} ---")
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        """ Process only the FIRST tool call - force one tool per iteration """
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found!")

        observation = tool_to_use(**tool_args)

        print(f"    [Tool Result]: {observation}")

        """ Providing agent feedback """
        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})
    print(f"MAX NUMBER OF ITERACTIONS reached without a final answer")
    return None


if __name__ == "__main__":
    print(f"Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
