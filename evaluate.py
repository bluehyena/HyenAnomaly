def evaluate(generator, classifier, num_samples):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, 512).to(device)
        samples = generator(z)
        _, _, labels = generator.classify(samples)
        labels = labels.argmax(dim=1).cpu().numpy()
        samples = samples.cpu().numpy()
    return samples, labels