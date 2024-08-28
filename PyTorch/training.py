
import matplotlib.pyplot as plt
import seaborn as sns

import nn_model as BC
import torch
import torch.nn as nn

import torch.optim as optim
import copy

from sklearn.model_selection import train_test_split

class modelTrain():
    def model_data_splits(df,feats,target):
        #inputs
        inputdf = df[feats].copy()
        x = torch.Tensor(list(inputdf.values)).T
        x = x.reshape(x.size(1), x.size(0))

        # Target
        y = torch.Tensor(df[target].values)

        # split the data into training and test sets (20% test)
        x_train_temp, x_test, y_train_temp, y_test = train_test_split(x.numpy(),y.numpy(), test_size=.2)
        # split into a validation set for learning curve (20% test)
        x_train, x_validation, y_train, y_validation = train_test_split(x_train_temp, y_train_temp, test_size=.2)

        # Convert the numpy arrays back to tensors
        x_train = torch.Tensor(x_train)
        x_test = torch.Tensor(x_test)
        x_validation = torch.Tensor(x_validation)

        y_train = torch.Tensor(y_train).unsqueeze(1)
        y_test = torch.Tensor(y_test).unsqueeze(1)
        y_validation = torch.Tensor(y_validation).unsqueeze(1)

        data_splits = {
            'x_train': x_train,
            'x_test': x_test,
            'x_validation': x_validation,
            'y_train': y_train,
            'y_test': y_test,
            'y_validation': y_validation
        }
        
        return data_splits

    def model_train(data_splits,steps=5000,learning_rate=1e-4,weight_decay=0,hidden_size=14,opt='adam',patience=20,verbose=True):
        num_features = data_splits['x_train'].size(1)
        deep_model = BC.BinaryClassification(input_size=num_features, hidden_size=hidden_size)

        # for binary awarness questions use the Binary Cross Entropy loss function to optimize model by adjusting weights during model training
        criterion = nn.BCELoss()

        optimizers={'adam':optim.Adam(deep_model.parameters(), lr=learning_rate,weight_decay=weight_decay),
                    'sgd':optim.SGD(deep_model.parameters(), lr=learning_rate,weight_decay=weight_decay)
                    }
        
        optimizer = optimizers.get(opt)

        losses=[]
        validation_losses=[]
        loss_steps=[]

        best_loss=float('inf')

        # training model
        for step in range(steps):
            deep_model.train()

            # zero gradients for every batch
            optimizer.zero_grad()

            #make predictions
            output = deep_model(data_splits['x_train'])
            
            # Compute the loss and its gradients
            loss = criterion(output, data_splits['y_train'])
            loss.backward()

            #adjust learning weights
            optimizer.step()

            # eval mode
            deep_model.eval()
            with torch.no_grad(): #pause gradient calc during validation
                val_output = deep_model(data_splits['x_validation'])
                val_loss = criterion(val_output, data_splits['y_validation'])
                                
                if verbose:
                    if (step + 1) % 100 == 0:
                        loss_steps.append(step)
                        losses.append(loss.item())
                        validation_losses.append(val_loss.item())
                        # loss and accuracy
                        val_y_pred = val_output.squeeze().round().long()
                        validation_accuracy= (val_y_pred == data_splits['y_validation']).float().mean()
                        print(f"step [{step+1}/{steps}],training loss: {round(loss.item(), 4)}, validation loss: {round(val_loss.item(),4)}, validation accuracy: {round(validation_accuracy.item(),4)}")

            # early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(deep_model.state_dict())  # Deep copy here      
                patience = patience  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    print('EARLY STOPPING...')
                    break
                        
        # Load the best model weights
        deep_model.load_state_dict(best_model_weights)

        losses={'train':losses,
                'validation':validation_losses,
                'steps':loss_steps}
        
        return deep_model,criterion,losses

    def model_eval(deep_model,data_splits,criterion):
        # evaluate
        deep_model.eval()
        with torch.no_grad():
            test_output = deep_model(data_splits['x_test'])
            # loss and accuracy
            test_loss = criterion(test_output, data_splits['y_test'])
            test_y_pred = test_output.squeeze().round().long()
            test_accuracy= (test_y_pred == data_splits['y_test']).float().mean()

        print(f"Total Test Loss: {round(test_loss.item(),4)}")
        print(f'Accuracy of the network: {100 * test_accuracy} %')
        return test_y_pred
    def plot_learning_curve(losses):
        fig, axes = plt.subplots(figsize=(15,5))
        fig.suptitle('Learning Curve')

        sns.lineplot(x=losses['steps'], y=losses['train'],ax=axes,label='train')
        sns.lineplot(x=losses['steps'], y=losses['validation'],ax=axes,label='validation')
        plt.show()