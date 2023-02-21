from kafka import KafkaProducer
import json
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

# Load CIFAR10 data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

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