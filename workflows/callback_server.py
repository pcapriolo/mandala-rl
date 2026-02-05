"""
Simple Flask server to receive workflow results via callback.

Receives training examples from distributed self-play and saves them locally.
"""
from flask import Flask, request, jsonify
import json
import pickle
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# Directory to save results
RESULTS_DIR = Path("data/distributed_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@app.route('/callback/results', methods=['POST'])
def receive_results():
    """Receive workflow results from Flyte callback."""
    try:
        data = request.json
        print("\n" + "="*60)
        print("RECEIVED CALLBACK FROM FLYTE")
        print("="*60)

        execution_id = data.get('executionId', 'unknown')
        status = data.get('status', 'unknown')
        outputs = data.get('outputs', {})

        print(f"Execution ID: {execution_id}")
        print(f"Status: {status}")
        print(f"Started: {data.get('startedAt')}")
        print(f"Completed: {data.get('completedAt')}")
        print(f"Duration: {data.get('durationMs', 0) / 1000:.1f}s")

        if status == 'succeeded':
            # Extract results
            num_games = outputs.get('num_games', 0)
            num_examples = outputs.get('num_examples', 0)
            examples = outputs.get('examples', [])

            print(f"\nResults:")
            print(f"  Games generated: {num_games}")
            print(f"  Training examples: {num_examples}")

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = RESULTS_DIR / f"examples_{execution_id}_{timestamp}.pkl"

            # Convert back to numpy arrays for training
            training_examples = []
            for ex in examples:
                training_examples.append((
                    ex['state'],   # Will be converted to tensor during training
                    ex['policy'],
                    ex['value']
                ))

            with open(result_file, 'wb') as f:
                pickle.dump(training_examples, f)

            print(f"\nSaved {len(training_examples)} examples to:")
            print(f"  {result_file}")

            # Also save raw JSON for inspection
            json_file = RESULTS_DIR / f"callback_{execution_id}_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"Saved full callback data to:")
            print(f"  {json_file}")

        elif status == 'failed':
            error = data.get('error', 'Unknown error')
            print(f"\nWorkflow FAILED: {error}")

            # Save error for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = RESULTS_DIR / f"error_{execution_id}_{timestamp}.json"
            with open(error_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"Saved error data to: {error_file}")

        print("="*60 + "\n")

        # Return 200 OK (required within 30 seconds)
        return jsonify({"status": "received"}), 200

    except Exception as e:
        print(f"Error processing callback: {e}")
        import traceback
        traceback.print_exc()
        # Still return 200 to acknowledge receipt
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    print("Starting callback server...")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print("\nEndpoints:")
    print("  POST /callback/results - Receive workflow results")
    print("  GET  /health          - Health check")
    print("\nListening on http://localhost:5000")
    print("Use ngrok to expose: ngrok http 5000")
    print()

    app.run(host='0.0.0.0', port=5000, debug=True)
