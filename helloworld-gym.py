import gym
from gym import wrappers
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.utils import to_categorical
import numpy as np
import os


def log(title, value=''):
    print(str(title) + ': ' + str(value))


experiment_path = '/tmp/cartpole-experiment'
model_path = 'helloworld-gym.h5'

env = gym.make('CartPole-v0')
# env = wrappers.Monitor(env, experiment_path, force=True)

if os.path.exists(model_path):
    log('===> Loading model...',)
    model = load_model(model_path)
else:
    log('===> Creating model...')
    model = Sequential()
    model.add(Dense(units=10, input_dim=4))
    model.add(Activation('tanh'))
    model.add(Dense(units=50))
    model.add(Activation('tanh'))
    model.add(Dense(units=10))
    model.add(Activation('tanh'))
    model.add(Dense(units=2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

STEPS = 11
EPISODES = 1000

epsilon = 0.33
training = STEPS

for i_episode in range(EPISODES):
    observation = env.reset()
    action = env.action_space.sample()

    x = []
    y = []
    for t in range(100):
        env.render()

        X = np.array([observation], ndmin=2)
        prediction = model.predict(X)
        action = np.argmax(prediction)
        observation, reward, done, info = env.step(action)

        # log('', '')
        # log('Prediction', prediction)
        # log('Action: ', action)
        # log('Observation: ', observation)
        # log('Reward: ', reward)
        # log('Done: ', done)
        # log('Info: ', info)

        x.append(observation)
        y.append(action)

        if done:
            log('Summary', "Episode {} finished after {} timesteps".format(
                i_episode, t + 1))
            if t > training:
                log('Training...')
                training += 1
                X = np.array(x, ndmin=2)
                Y = to_categorical(np.array(y), 2)
                model.train_on_batch(X, Y)
            break

log('Saving model')
model.save(model_path)

# gym.upload(experiment_path, api_key='sk_MVMO22OSxGRr4vsZS1rWg')
