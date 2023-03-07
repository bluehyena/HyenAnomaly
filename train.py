import torch
import torch.nn as nn
import numpy as np
import torchvision.utils as vutil

def train(generator, discriminator, classifier, dataloader, num_epochs, lr, betas):
    # 손실 함수
    adversarial_loss = nn.BCEWithLogitsLoss()
    class_loss = nn.CrossEntropyLoss()
    
    # 옵티마이저
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    c_optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=betas)
    
    # PGGAN 모델 객체 생성
    model = PGGAN(generator, discriminator, classifier)
    model = model.to(device)
    
    # 학습 시작
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            
            # Discriminator 학습
            d_optimizer.zero_grad()
            
            # 진짜 데이터에 대한 판별 결과
            real_score = model.discriminator(images)
            real_labels = torch.ones(batch_size).to(device)
            real_loss = adversarial_loss(real_score, real_labels)
            
            # 가짜 데이터에 대한 판별 결과
            z = torch.randn(batch_size, 512).to(device)
            fake = model.generator(z)
            fake_score = model.discriminator(fake.detach())
            fake_labels = torch.zeros(batch_size).to(device)
            fake_loss = adversarial_loss(fake_score, fake_labels)
            
            # 판별자의 전체 손실
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Generator 학습
            g_optimizer.zero_grad()
            
            # 가짜 데이터에 대한 판별 결과
            fake_score = model.discriminator(fake)
            real_labels = torch.ones(batch_size).to(device)
            
            # Generator의 전체 손실
            g_loss = adversarial_loss(fake_score, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            # Classifier 학습
            c_optimizer.zero_grad()
            
            # Generator로 생성한 가짜 데이터에 대한 분류 결과
            _, _, fake_labels = model.forward(fake, alpha=1)
            
            # Classifier의 전체 손실
            c_loss = class_loss(fake_labels, labels)
            c_loss.backward()
            c_optimizer.step()
            
            # 출력
            batches_done = epoch * len(dataloader) + i
            if batches_done % 100 == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [C loss: %f]"
                      % (epoch+1, num_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item(), c_loss.item()))
                
        # 한 단계 해상도를 높입니다.
        if epoch < num_epochs-1:
            model.generator.add_resolution()
            model.discriminator.add_resolution()
            
    return model
