from env_1 import SimpleEnv
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import copy

class CircleAgent(object):
    """
    docstring
    """
    def __init__(self, env=None):
        self.r = 0.1
        self.step = 0.05 * np.pi
        self.cx = 0.45
        self.cy = 0.52

        self.noise = 0.01
        self.start_noise = 0.1
        self.env = env       

    
    def action(self, state):
        cur_pos = np.array([state[0] - self.cx, state[2] - self.cy])
        cur_the = np.arctan2(cur_pos[1], cur_pos[0])
        next_the = cur_the
        # if abs(np.sqrt(np.sum(np.square(cur_the))) - self.r) < 0.1:
        next_the += self.step
        next_pos = np.array([self.cx + self.r * np.cos(next_the), 0, self.cy + self.r * np.sin(next_the)])
        # print(next_the)
        dpos = next_pos - state
        return dpos

    def toLocationAround(self, state):
        dpos = self.noise * np.random.randn(3)
        return dpos

    def pickRandomStartLocation(self):
        theta = 2 * np.pi * np.random.uniform()
        pho = self.r  #+ self.start_noise * np.random.randn()

        target_ee = np.array([self.cx + pho * np.cos(theta), 0, self.cy + pho * np.sin(theta)])
        state = self.env.reset()
        while (np.sqrt(np.sum(np.square(target_ee - state['ee_position']))) > 0.05):
            action =  target_ee - state['ee_position']
            state, reward, done, info = env.step(action)
        print("reach location", target_ee)


        



def test_circle_agent():
    # test circle agent
    s = np.array([-0.1, 0, 0])
    ag = CircleAgent(s)
    r = [s]
    for i in range(5):
        dpos = ag.action(s)        
        # print(dpos)
        r.append(np.array(s + dpos))
        s = s + dpos #+ 0.01*np.random.randn(*s.shape)
    data = np.array(r)
    plt.scatter(data[:, 0], data[:, 2])
    plt.show()



# test_circle_agent()
env = SimpleEnv()


ag = CircleAgent(env)

# to another location to start

# state_pair = np.zeros(6)
# state_pair[:3] = state['ee_position']
record = []
state = env.reset()
for i in range(400):
    # state = env.reset()
    # ag.pickRandomStartLocation()
    start_time = time.time()
    curr_time = time.time() - start_time
    
    step = 0
    traj = []

    st = i % 100

    while step < (20000 + st):        
        curr_time = time.time() - start_time
        
        # state_pair = np.zeros(6)
        # state_pair[:3] = state['ee_position']
        action = ag.action(state['ee_position'])
        state, reward, done, info = env.step(action)
        # state_pair[3:] = state['ee_position']
        # record.append(state_pair)
        if step % 100 == st:
            # print(curr_time, step)
            traj.append(state['ee_position'])
        step = step + 1

    record.append(copy.deepcopy(traj))
    # print(state['ee_position'])
    # img = env.render()
    if done:
        break
env.close()
data = np.array(record)
print(data.shape)
pickle.dump(data, open('/home/jingjia/iliad/logs/data/benchmark-circle-100.pkl', 'wb'))
# plt.scatter(data[:, 0], data[:, 2])
# plt.axis('square')
# plt.show()

data = pickle.load(open('/home/jingjia/iliad/logs/data/benchmark-circle-100.pkl', 'rb'))
m = len(data)
for i in range(m):
    plt.scatter(data[i, :, 0], data[i, :, 2])
plt.axis('square')
plt.show()