from trainer_vae import Trainer

trainer = Trainer()
autoenc = trainer.train(epochs=50)
trainer.show_real_gen_data_comparison(autoenc, load_model=True, save=True)