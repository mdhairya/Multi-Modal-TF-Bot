import tensorflow as tf
import configparser
from models.vqa_model import VQAModel

def train_step(model, images, questions, answers, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        predictions = model((images, questions), training=True)
        loss = loss_fn(answers, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_acc_metric.update_state(answers, predictions)
    return loss

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Initialize Model, Optimizer, and Metrics
    model = VQAModel(config)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=config.getfloat('TRAINING', 'learning_rate'))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # NOTE: Replace with your actual tf.data.Dataset loader logic
    # dataset = load_vqa_dataset(config) 
    
    epochs = config.getint('TRAINING', 'epochs')
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # For demonstration, assuming `dataset` yields (images, questions, answers)
        # for step, (images, questions, answers) in enumerate(dataset):
        #     loss = train_step(model, images, questions, answers, optimizer, loss_fn, train_acc_metric)
        #     if step % 50 == 0:
        #         print(f"Step {step}: Loss = {float(loss):.4f}, Accuracy = {float(train_acc_metric.result()):.4f}")
        
        train_acc_metric.reset_state()
        
    # Save the weights
    # model.save_weights('./checkpoints/vqa_model_weights')

if __name__ == "__main__":
    main()
