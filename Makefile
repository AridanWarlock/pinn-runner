.PHONY: build-dev build-prod run-dev run-prod

build-dev:
	docker build -f Dockerfile.dev -t pinn-runner-image .

build-prod:
	docker build -t pinn-runner-image .

run-dev:
	docker run -it \
		-v $(PWD):/app \
		-v $(PWD)/dev/task_data:/task_data \
		-v $(PWD)/dev/task_output:/task_output \
		pinn-runner-image bash

run-prod:
	docker run -d pinn-runner