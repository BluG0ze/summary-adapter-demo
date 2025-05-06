import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.prompts import SYSTEM_MSG, USER_PROMPT_PREFIX


BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
PEFT_MODEL_PATH = "../dpo-rola/dpo_lr1e6_bz8_beta03_fsdp_3ep/checkpoint-300"


def preprocess_prompt(text):
    """
    Converts the input text into a conversation format by adding system message and prompt prefix.
    Args:
        text (str): The input text to be summarized.
    Returns:
        list: conversation containing system message and user prompt.
    """
    conversation = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_PROMPT_PREFIX + text},
    ]
    return conversation


def generate_summary(text):
    """
    1. Preprocess the input text by adding system message and user prompt prefix.
    2. Tokenize the conversation using the tokenizer.
    3. Generate a summary using the model and adapter.
    4. Decode the generated text to get the final summary.
    Args:
        text (str): The input text to be summarized.
    Returns:
        str: The generated summary.
    """
    conversation = preprocess_prompt(text)
    # Wrap with chat template and tokenize the conversation
    inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt", truncation=True, max_length=32768, add_generation_prompt=True).to(model.device)
    attention_mask = torch.ones_like(inputs)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            do_sample=True,
            max_length=32768, #32k
            temperature=0.7,
            top_p=0.9,
        )
    # Extract the generated text from the model output
    input_length = inputs.shape[1]
    completion = outputs[0][input_length:]
    # Decode the generated text
    generated_text = tokenizer.decode(completion, skip_special_tokens=True)
    return generated_text


def sub_thread_chatbot(user_input):
    """
    This function is run in a separate thread to handle the chatbot interaction.
    It generates a summary based on the user input and prints it.
    Args:
        user_input (str): The text input from the user to be summarized.
    """
    try:
        print("generating...\n")
        summary = generate_summary(user_input)
        print(f"Here is the summary:\n{summary}\n\n")
    except Exception as e:
        print(f"Error: {e}")


def naive_chat_bot():
    """
    A simple chatbot that takes user input and generates a summary using the model.
    It runs in a loop until the user types 'exit'.
    """
    while True:
        user_input = input("Please input the text you want to summerize (or type 'exit' to quit): \n")
        print("\n")

        if user_input.lower() == "exit":
            print("Bye!")
            break

        # Create a new thread for get the summary
        thread = threading.Thread(target=sub_thread_chatbot, args=(user_input,))
        thread.start()
        thread.join()


def example_usage():
    """
    Example usage of the generate the summary given a text input.
    """
    # Example usage
    user_input = "在這幾個月的時間內，這個變種病毒如何從不存在，變成了英格蘭部分地區最常見的變種病毒呢？英國政府的專家顧問說，他們有「高度把握」，認為這個變種病毒「比其他病毒變種更容易傳播」。 雖然，有關變種病毒的檢測都處於早期階段，包含很多的不確定性和未解決的問題。但正如之前我所說過的，病毒總是一直發生變異，因此重要的一點是，我們必須始終關注病毒的變異行為。 為什麼變種病毒引起擔憂？ 這三件事合在一起，這個變種病毒引起了高度關注： 這些因素導致產生了一種傳播力更強的變種新冠病毒。 但是，目前我們尚未對這新變種病毒有完整的掌握。 有時候，只是因為時機或地點的配合，譬如在倫敦，該病毒便快速的傳播。原本倫敦之前處於二級管制，但因為該變種病毒迅速傳播，倫敦馬上進入四級管制。 「實驗室當然能檢測並分析這個病毒，但您還能夠等上數個星期或數個月（等待研究結果後再去限制病毒傳播的速度嗎？）。現在的情況恐怕不允許我們這麼做。」英國國家基因組 (Genomics UK)新冠研究處尼克羅曼（Nick Loman）教授告訴記者。 它傳播的速度有多快？ 該變種病毒最早在2020年9月被發現。到了11月，倫敦大約有四分之一的新冠感染與該變種病毒有關。 12月中旬，這一數字已接近三分之二。 在米爾頓凱恩斯燈塔實驗室（Milton Keynes Lighthouse Laboratory）等機構的測試結果中，可以看到該變種病毒如何一步步影響了統計結果。 數學家們一直在統計和比較不同變種的數據，以掌握單一變種病毒的特點。 但要搞清楚變種是源於人類行為還是病毒本身並不容易。 英國首相約翰遜（Boris Johnson）提到，該變種病毒的傳播力增加幅度可高達70％。這可能會使R值（表示大流行是在增長還是在縮小規模的數值）增加0.4。 事實上，70%這個估算，先出現在上周五（12月18日）倫敦帝國理工學院的沃爾茲（Erik Volz）博士的一個演講中。他說：「現在說出來還為時過早，但是從目前觀察來看，它的增加非常快，速度超過了以前的新冠變種病毒，請務必注意這個現象。」 不過，至今我們仍沒有這個變種病毒傳染力高了多少的「精凖」數字。尚未公開其研究結果的科學家告訴我，目前研究數據有的高於70%，有的則低於該數字。 但對於該變種病毒是否具有更高傳染性，疑問仍在。 諾丁漢大學病毒學家鮑爾（Jonathan Ball）教授說：「目前出現在公眾視野的大量證據，仍無法就該病毒是否加快大流行傳播速度，提出有力及令人信服的意見。」 它傳播了多遠？ 分析認為，該變種病毒可能是從英國的病人中出現，或來自對於監測冠狀病毒突變能力較低的國家或地區。 除了北愛爾蘭，該變種病毒遍布英國各地，但主要集中在倫敦和英格蘭東部、東南部。 英國其他地方似乎尚未大規模發現該病毒。 一直在監測世界各地病毒樣本遺傳密碼的機構Nextstrain說，數據表明，丹麥、澳大利亞的變種病毒病例都來自英國。荷蘭也報告了相同確診病例。 在南非出現的一個類似的變種病毒與它有些相同的突變，但似乎與其無關。 病毒變種以前發生過嗎？ 是的。 在中國武漢首次發現的新冠病毒，與現在在世界大多數角落髮現的病毒不同。今年2月在歐洲出現的「D614G」變種病毒，是目前全球主要新冠病毒確診的種類。 另一個變種病毒「A222V」遍布歐洲，與今年夏天在西班牙的渡假者有關。 我們對病毒突變了解多少？ 目前已經發佈的初步分析顯示，該變種病毒發生了17個潛在的重要改變。其中，病毒的棘蛋白(spike protein)發生了變化，這是病毒用來「解鎖」進入人體細胞的鑰匙。 一個名為「N501Y」的變異，改變了病毒棘狀物（spike）的最重要部分：受體結合域（receptor-binding domain），令它們更容易侵入人體細胞：因為這是病毒棘首先與人體細胞表面接觸的地方。任何使病毒更容易進入人體細胞的變動，都可能增強病毒的毒性。 羅曼教授說：「看起來就像是一次重要的重組改裝。」 另一種被稱為H69 / V70的變異中，病毒一小部分棘已經被去除，這種變化之前出現了好幾次，包括之前有關貂被感染病毒的案例。 劍橋大學教授古帕塔（Ravi Gupta）說，實驗證明這種突變將感染力提高了兩倍。同一小組的研究也證實，這變體會降低曾經的感染者血液​​中抗體抵禦病毒的效力。他又告訴BBC：「相關變化正迅速增加，這令政府感到擔憂。我們及多數科學家也同樣憂慮。」 變種病毒來自哪裏？ 該變種病毒變異程度不尋常的高。 目前最有可能的解釋是，這種變異出現在無法抵抗病毒和免疫系統較弱的患者中，後者的身體成為病毒變異的溫牀。 它更加致命嗎？ 儘管有必要繼續監控，但目前尚無證據證明該病毒更加致命。 但是，僅僅增加傳染數就足以給醫院造成麻煩，因為如果新的變種病毒繼續蔓延，意味著更多人會被更快感染，也意味者更多人需要住院治療。 目前三種領先的疫苗都對現有的變種病毒有效。 疫苗能對抗變種病毒嗎？ 答案幾乎是肯定的！至少目前如此。 目前3種領先的疫苗都對現有的變種病毒有效。疫苗訓練免疫系統攻擊病毒的多個不同部分，因此即時該變種病毒部分已開始突變，疫苗仍起作用。 古帕教授說：「但是，如果病毒的突變越來越多，你就要開始擔心了。」 「因為，這表示該變種病毒可能正往『疫苗逃逸』（vaccine escape）的路上前進，它已向此邁出了幾步。」 當病毒變異到能避開疫苗的全部作用，並繼續傳染人時，就是「疫苗逃逸」。這是該變種病毒現在最令人擔憂的地方。此外，該變種病毒只是最新變種的一種，這表示病毒正在變化調整，同時感染越來越多的人。 格拉斯哥大學教授羅伯遜（David Robertson）在12月18日的談話總結說：「該變種病毒很可能產生具備『疫苗逃逸』功能的突變體。」這便意味著，我們將處於面對類似於流感的情況，需要定期更新疫苗。 幸運的是，我們現有的疫苗也很容易調整以面對病毒變種 。"
    summary = generate_summary(user_input)
    print(f"Here is the summary:\n{summary}\n\n")


if __name__ == "__main__":
    # init model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto")
    model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH, device_map="auto")
    model.eval()

    print(f"Successfully loaded Model and Adapter on device: {model.device}")

    # Uncomment the following line to run the chatbot
    naive_chat_bot()
    #example_usage()
