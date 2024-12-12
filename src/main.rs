use langchain_rust::{
    chain::{builder::ConversationalChainBuilder, Chain, ConversationalChain},
    llm::client::Ollama,
    prompt_args,
};
use std::sync::Arc;
use teloxide::{
    dispatching::UpdateFilterExt,
    dptree,
    prelude::*,
    types::{ChatAction, Message, Update},
    Bot,
};

const LLM_MODEL: &str = "qwen2.5-coder:32b";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();

    let ollama = Ollama::default().with_model(LLM_MODEL);
    let chain = Arc::new(ConversationalChainBuilder::new().llm(ollama).build()?);

    let bot = teloxide::Bot::from_env();
    let handler = Update::filter_message().endpoint(message_handler);

    Dispatcher::builder(bot.clone(), handler)
        .dependencies(dptree::deps![chain])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;

    Ok(())
}

async fn message_handler(
    bot: Bot,
    chain: Arc<ConversationalChain>,
    msg: Message,
) -> anyhow::Result<()> {
    if let Some(text) = msg.text() {
        bot.send_chat_action(msg.chat.id, ChatAction::Typing)
            .await?;
        let input_variables = prompt_args! { "input" => text };
        let result = chain.invoke(input_variables).await?;
        bot.send_message(msg.chat.id, result).await?;
    }
    Ok(())
}
