.PHONY: setup install run dev test demo clean

# Setup Ollama and pull model
setup-ollama:
	@echo "Installing Ollama..."
	brew install ollama || true
	@echo "Starting Ollama service..."
	ollama serve &
	sleep 3
	@echo "Pulling Mistral model..."
	ollama pull mistral
	@echo "✅ Ollama setup complete!"

# Install Python dependencies
install:
	python -m venv venv || true
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Run production server
run:
	. venv/bin/activate && uvicorn src.automotive_intent.app:app --host 0.0.0.0 --port 8000

# Run development server (auto-reload)
dev:
	. venv/bin/activate && uvicorn src.automotive_intent.app:app --reload

# Run without NLLB (faster startup)
dev-fast:
	. venv/bin/activate && AMI_USE_NLLB=false uvicorn src.automotive_intent.app:app --reload

# Run tests
test:
	. venv/bin/activate && pytest tests/ -v

# Run Streamlit demo
demo:
	. venv/bin/activate && streamlit run ui/streamlit_app.py

# Clean cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache 2>/dev/null || true

# Full setup (first time)
full-setup: setup-ollama install
	@echo ""
	@echo "✅ Full setup complete!"
	@echo ""
	@echo "To start the API:  make run"
	@echo "To start the demo: make demo"
