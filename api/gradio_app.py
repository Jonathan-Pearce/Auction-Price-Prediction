# =============================================================================
# Auction Price Prediction - Gradio Interface
# =============================================================================
"""
Gradio web interface for auction price prediction.

This provides a simple UI for users to:
1. Enter a MaxSold item URL
2. Get a price prediction with confidence interval
3. View individual model predictions

Deployment:
- Local: python -m api.gradio_app
- HF Spaces: This file serves as the main app
"""

from typing import Any

from loguru import logger

# Gradio import (may not be installed in all environments)
try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

from src.config import settings


# =============================================================================
# Prediction Function
# =============================================================================


def predict_price(url: str) -> dict[str, Any]:
    """
    Predict winning price for a MaxSold item URL.

    Args:
        url: MaxSold item URL

    Returns:
        Dictionary with prediction results
    """
    if not url or not url.strip():
        return {
            "error": "Please enter a valid MaxSold item URL",
            "predicted_price": None,
            "confidence_interval": None,
            "model_predictions": None,
        }

    try:
        logger.info(f"Gradio prediction request for: {url}")

        # TODO: Implement actual prediction
        # from src.modeling.predict import EnsemblePredictor
        # predictor = EnsemblePredictor()
        # result = predictor.predict_from_url(url)

        # Placeholder response
        return {
            "predicted_price": "$0.00",
            "confidence_interval": "$0.00 - $0.00",
            "model_predictions": {
                "Tabular Model": "Not loaded",
                "Image Model": "Not loaded",
                "Text Model": "Not loaded",
                "Bid Sequence Model": "Not loaded",
            },
            "status": "Models not yet trained - this is a placeholder",
        }

    except Exception as e:
        logger.error(f"Gradio prediction failed: {e}")
        return {
            "error": str(e),
            "predicted_price": None,
            "confidence_interval": None,
            "model_predictions": None,
        }


def format_prediction(result: dict[str, Any]) -> tuple[str, str, str]:
    """
    Format prediction result for Gradio output.

    Returns:
        Tuple of (price_text, confidence_text, details_text)
    """
    if "error" in result and result["error"]:
        return (
            f"❌ Error: {result['error']}",
            "",
            "",
        )

    price = result.get("predicted_price", "N/A")
    confidence = result.get("confidence_interval", "N/A")

    # Format model predictions
    model_preds = result.get("model_predictions", {})
    details_lines = ["### Individual Model Predictions\n"]
    for model, pred in model_preds.items():
        details_lines.append(f"- **{model}**: {pred}")

    if "status" in result:
        details_lines.append(f"\n⚠️ {result['status']}")

    return (
        f"## 💰 Predicted Price: {price}",
        f"**95% Confidence Interval:** {confidence}",
        "\n".join(details_lines),
    )


# =============================================================================
# Gradio Interface
# =============================================================================


def create_interface() -> "gr.Blocks":
    """Create the Gradio interface."""
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is not installed. Run: pip install gradio")

    with gr.Blocks(
        title="Auction Price Predictor",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown(
            """
            # 🔨 MaxSold Auction Price Predictor

            Predict the winning price for items in MaxSold online auctions.

            **How to use:**
            1. Find an item on [MaxSold.com](https://maxsold.com)
            2. Copy the item URL
            3. Paste it below and click "Predict Price"

            This model uses an ensemble of:
            - 📊 Tabular features (auction metadata, item category, etc.)
            - 🖼️ Image analysis (item photos)
            - 📝 Text analysis (item descriptions)
            - 📈 Bid sequence patterns (bidding history)
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                url_input = gr.Textbox(
                    label="MaxSold Item URL",
                    placeholder="https://maxsold.com/auction/12345/item/67890",
                    lines=1,
                )
                predict_btn = gr.Button("🔮 Predict Price", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### Example URLs
                    - Coming soon...
                    """
                )

        with gr.Row():
            with gr.Column():
                price_output = gr.Markdown(label="Predicted Price")
                confidence_output = gr.Markdown(label="Confidence")

        with gr.Accordion("Model Details", open=False):
            details_output = gr.Markdown()

        # Event handler
        def on_predict(url: str) -> tuple[str, str, str]:
            result = predict_price(url)
            return format_prediction(result)

        predict_btn.click(
            fn=on_predict,
            inputs=[url_input],
            outputs=[price_output, confidence_output, details_output],
        )

        url_input.submit(
            fn=on_predict,
            inputs=[url_input],
            outputs=[price_output, confidence_output, details_output],
        )

        gr.Markdown(
            """
            ---
            **About this project:**
            This is a machine learning project that predicts auction winning prices
            using multiple data modalities. Built with PyTorch, FastAPI, and Gradio.

            [GitHub Repository](https://github.com/Jonathan-Pearce/Auction-Price-Prediction-)
            """
        )

    return interface


# =============================================================================
# Entry Points
# =============================================================================


def launch_local(share: bool = False) -> None:
    """Launch Gradio interface locally."""
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio.server_port,
        share=share or settings.gradio.share,
    )


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch Gradio interface")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run on",
    )

    args = parser.parse_args()

    if args.port:
        settings.gradio.server_port = args.port

    launch_local(share=args.share)


# For HF Spaces: Create interface at module level
if GRADIO_AVAILABLE:
    demo = create_interface()

if __name__ == "__main__":
    main()
