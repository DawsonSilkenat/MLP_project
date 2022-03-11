import data_provider
import models 

# data = iter(data_provider.LanguageDataset(batch_size=20))

model = models.BiasDetector(5, models.BertEmbedding, num_layers=1)

train = data_provider.LanguageDataset("../bias_data/bias_data/WNC/biased.word.train", batch_size=20)
eval = data_provider.LanguageDataset("../bias_data/bias_data/WNC/biased.word.test", batch_size=20)

model.run_experiments(train, eval, num_epochs=2)


# "/bias_data/bias_data/WNC/biased.word.train"

# print(model.parameters())
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# bert = model.get_embedder()

# values, targets = next(data)

# targets = bert.update_targets(values, targets)
# # print(targets)

# results = model(values)
# print(results)

# print()
# print(targets.size())
# print(results.size())