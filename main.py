import trainer as T

def main():
    trainer = T.Trainer()
    for _ in range(20):
        trainer.collect_training_data(True, std = 0.5)
        p = 1
    for _ in range(1000000):
        trainer.single_train_step()

if __name__ == '__main__':
    main()
