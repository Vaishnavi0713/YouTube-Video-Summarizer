
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from transformers import pipeline
import gradio as gr
import os


# --------------------------------------------------
# 1. LOAD DISTILBART MODEL
# --------------------------------------------------

model_path = r"D:\Projects\models\distilbart-cnn-12-6"

text_summary = pipeline(
    "summarization",
    model=model_path,
    device=-1
)


# --------------------------------------------------
# 2. EXTRACT YOUTUBE VIDEO ID
# --------------------------------------------------

def extract_video_id(url):

    try:
        parsed_url = urlparse(url.strip())

        if parsed_url.hostname in ["youtu.be", "www.youtu.be"]:
            return parsed_url.path.lstrip("/")

        if parsed_url.hostname and "youtube.com" in parsed_url.hostname:

            if parsed_url.path == "/watch":
                return parse_qs(
                    parsed_url.query
                ).get("v", [None])[0]

            if parsed_url.path.startswith(("/embed/", "/shorts/")):
                return parsed_url.path.split("/")[2]

    except Exception:
        return None

    return None


# --------------------------------------------------
# 3. GENERATE SUMMARY USING CHUNKING
# --------------------------------------------------

def summary(input_text):

    max_chunk_tokens = 900

    input_ids = text_summary.tokenizer.encode(
        input_text,
        add_special_tokens=False
    )

    chunks = [
        input_ids[i:i + max_chunk_tokens]
        for i in range(0, len(input_ids), max_chunk_tokens)
    ]

    summaries = []

    print(f"\nTotal chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):

        print(
            f"Summarizing chunk {index + 1}/{len(chunks)}..."
        )

        chunk_text = text_summary.tokenizer.decode(
            chunk,
            skip_special_tokens=True
        )

        output = text_summary(
            chunk_text,
            max_length=130,
            min_length=25,
            do_sample=False,
            truncation=True
        )

        summaries.append(output[0]["summary_text"])

    combined_summary = " ".join(summaries)

    # Reduce combined summary if it is still too long
    while True:

        final_tokens = text_summary.tokenizer.encode(
            combined_summary,
            add_special_tokens=False
        )

        if len(final_tokens) <= max_chunk_tokens:
            break

        batches = [
            final_tokens[i:i + max_chunk_tokens]
            for i in range(0, len(final_tokens), max_chunk_tokens)
        ]

        new_summaries = []

        for batch in batches:

            batch_text = text_summary.tokenizer.decode(
                batch,
                skip_special_tokens=True
            )

            output = text_summary(
                batch_text,
                max_length=130,
                min_length=25,
                do_sample=False,
                truncation=True
            )

            new_summaries.append(output[0]["summary_text"])

        new_combined = " ".join(new_summaries)

        if new_combined.strip() == combined_summary.strip():
            break

        combined_summary = new_combined

    final_output = text_summary(
        combined_summary,
        max_length=180,
        min_length=40,
        do_sample=False,
        truncation=True
    )

    return final_output[0]["summary_text"]


# --------------------------------------------------
# 4. FETCH TRANSCRIPT AND GENERATE SUMMARY
# --------------------------------------------------

def process_video(video_url):

    if not video_url or not video_url.strip():
        raise gr.Error("Please enter a YouTube video URL.")

    video_id = extract_video_id(video_url)

    if not video_id:
        raise gr.Error("Invalid YouTube URL. Please check the link.")

    try:

        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id)

        transcript = " ".join(
            snippet.text for snippet in transcript_data
        )

        if not transcript.strip():
            raise gr.Error("The video transcript is empty.")

        print("\nTranscript extracted successfully.")
        print("\nGenerating summary...")

        video_summary = summary(transcript)

        # Save transcript
        transcript_path = os.path.abspath("transcript.txt")

        with open(
            transcript_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(transcript)

        # Save summary
        summary_path = os.path.abspath("summary.txt")

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(video_summary)

        print("\nSummary generated successfully.")
        print("Summary file:", summary_path)
        print("Transcript file:", transcript_path)

        # Return text and actual file paths to Gradio
        return (
            video_summary,
            transcript,
            summary_path,
            transcript_path
        )

    except gr.Error:
        raise

    except Exception as error:

        print("Error:", error)

        raise gr.Error(
            f"Unable to process this video: {error}"
        )


# --------------------------------------------------
# 5. CLEAR ALL FIELDS
# --------------------------------------------------

def clear_all():

    return (
        "",
        "",
        None,
        None
    )


# --------------------------------------------------
# 6. CUSTOM INTERFACE DESIGN
# --------------------------------------------------

custom_css = """

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 18px 3% !important;
    background: #151719 !important;
    color: #f5f5f5 !important;
    font-family: Arial, sans-serif !important;
}

#main-title {
    text-align: center;
    margin: 5px 0 22px 0;
    color: #ffffff;
    font-size: 32px;
    font-weight: 700;
}

#url-section {
    background: #202326;
    border: 1px solid #44494e;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 18px;
}

.output-card {
    background: #202326;
    border: 1px solid #44494e;
    border-radius: 12px;
    padding: 14px;
    height: 100%;
}

.section-heading {
    color: #ff4242 !important;
    font-size: 21px !important;
    font-weight: 700 !important;
    margin-bottom: 10px !important;
}

textarea,
input {
    background: #292d31 !important;
    color: #ffffff !important;
    border: 1px solid #646a70 !important;
    border-radius: 9px !important;
    font-size: 15px !important;
}

textarea::placeholder,
input::placeholder {
    color: #aeb3b8 !important;
}

button {
    border-radius: 9px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    min-height: 48px !important;
    transition: 0.2s ease !important;
}

#summarize-btn,
#download-summary,
#download-transcript {
    background: #e53935 !important;
    color: white !important;
    border: 1px solid #e53935 !important;
}

#summarize-btn:hover,
#download-summary:hover,
#download-transcript:hover {
    background: #c62828 !important;
}

#clear-btn {
    background: transparent !important;
    color: white !important;
    border: 1px solid #777777 !important;
    width: 100% !important;
}

#clear-btn:hover {
    background: #292d31 !important;
}

footer {
    display: none !important;
}

@media (max-width: 768px) {

    .gradio-container {
        padding: 12px !important;
    }

    #main-title {
        font-size: 23px;
    }

    .section-heading {
        font-size: 18px !important;
    }
}

"""


# --------------------------------------------------
# 7. BUILD GRADIO INTERFACE
# --------------------------------------------------

with gr.Blocks(
    css=custom_css,
    title="YouTube Video Summarizer",
    theme=gr.themes.Base(
        primary_hue="red",
        secondary_hue="gray",
        neutral_hue="slate"
    )
) as demo:

    gr.Markdown(
        "# ▶ YouTube Video Summarizer",
        elem_id="main-title"
    )

    # YouTube URL and summarize button

    with gr.Row(elem_id="url-section"):

        youtube_url = gr.Textbox(
            placeholder="Paste YouTube URL here...",
            label="",
            lines=1,
            scale=5,
            container=False
        )

        summarize_btn = gr.Button(
            "▶ Summarize Video",
            variant="primary",
            scale=1,
            elem_id="summarize-btn"
        )

    # Summary and transcript side by side

    with gr.Row(equal_height=True):

        # Generated summary

        with gr.Column(
            scale=1,
            elem_classes=["output-card"]
        ):

            gr.Markdown(
                "### ▤ Generated Summary",
                elem_classes=["section-heading"]
            )

            summary_output = gr.Textbox(
                placeholder="The generated summary will appear here...",
                show_label=False,
                lines=12,
                max_lines=12,
                interactive=False
            )

            summary_download = gr.DownloadButton(
                "⬇ Download Summary",
                variant="secondary",
                interactive=True,
                elem_id="download-summary"
            )

        # Complete transcript

        with gr.Column(
            scale=1,
            elem_classes=["output-card"]
        ):

            gr.Markdown(
                "### ▤ Complete Video Transcript",
                elem_classes=["section-heading"]
            )

            transcript_output = gr.Textbox(
                placeholder="The complete video transcript will appear here...",
                show_label=False,
                lines=12,
                max_lines=12,
                interactive=False
            )

            transcript_download = gr.DownloadButton(
                "⬇ Download Transcript",
                variant="secondary",
                interactive=True,
                elem_id="download-transcript"
            )

    # Clear all button

    clear_btn = gr.Button(
        "▣ Clear All",
        elem_id="clear-btn"
    )

    # --------------------------------------------------
    # 8. BUTTON ACTIONS
    # --------------------------------------------------

    summarize_btn.click(
        fn=process_video,
        inputs=youtube_url,
        outputs=[
            summary_output,
            transcript_output,
            summary_download,
            transcript_download
        ]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[
            youtube_url,
            summary_output,
            transcript_output,
            summary_download,
            transcript_download
        ]
    )


# --------------------------------------------------
# 9. LAUNCH APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )
