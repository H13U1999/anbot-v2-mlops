build:
	docker build -t an-bot-v2 .
run:
	docker run -p 8080:5000 an-bot-v2