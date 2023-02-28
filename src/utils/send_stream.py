import json
import time
import numpy as np
from kafka import KafkaProducer
from utilities import load_data

# Load CIFAR10 data
trainloader, testloader = load_data()

# Define Kafka producer configuration
producer = KafkaProducer(bootstrap_servers='kafka-broker:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Send CIFAR10 data to Kafka
while True:
    for data, _ in testloader:
        time.sleep(1)
        data = data.numpy()
        message = {'data': data.tolist()}
        producer.send("CIFAR-10", 
                    'cifar10-data', value=message)

    producer.flush()
