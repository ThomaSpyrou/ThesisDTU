from kafka import KafkaProducer
import json

from utilities import load_data

# Load CIFAR10 data
trainloader, testloader = load_data()

# Define Kafka producer configuration
kafka_broker = 'localhost:9092'
producer_config = {'bootstrap_servers': [kafka_broker]}

# Create Kafka producer
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), **producer_config)

# Send CIFAR10 data to Kafka
for data, _ in testloader:
    data = data.numpy()
    message = {'data': data.tolist()}
    print(message)
    producer.send('cifar10-data', value=message)

producer.flush()