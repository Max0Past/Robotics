import numpy as np
import tqdm
import gymnasium as gym


# Action constants.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

def play_episode(env, policy_iteration, render=False, iter_max=1000):
    done = False
    obs, _ = env.reset()
    total_reward = 0
    i_iter = 0
    while not done:
        action = policy_iteration.pick_action(obs)
        obs, rew, done, _, _ = env.step(action)
        if render:
            env.render()
        total_reward += rew

        if iter_max < i_iter:
            done = True
        i_iter += 1
    return total_reward


#######################################################
### Task: Implement Policy Iteration algorithm.     ###
### Завдання: Імплементувати ітерацію стратегіями   ###
#######################################################
class PolicyIteration:
    # Додано параметр `mode` 
    def __init__(self, transition_probs, states, actions, mode='deterministic'):
        self.transition_probs = transition_probs
        self.states = states
        self.actions = actions
        self.policy = np.ones([len(self.states), len(self.actions)]) / len(self.actions)
        # Зберігаємо обраний режим
        self.mode = mode

    def pick_action(self, obs):
        ### Using a policy pick an action for the state `obs`
        ### Використовуючи стратегію обрати дію для стану `obs`
        
        # Логіка вибору дії залежить від режиму
        if self.mode == 'stochastic':
            # Режим з випадковістю: вибір дії відповідно до ймовірностей
            return np.random.choice(self.actions, p=self.policy[obs])
        else: # 'deterministic'
            # Детермінований режим: завжди обирати найкращу дію
            return np.argmax(self.policy[obs])

    def run(self):
        # Логіка навчання залишається незмінною, вона знаходить оптимальну стратегію
        theta = 1e-9
        discount_factor = 0.99
        value_function = np.zeros(len(self.states))

        while True:
            # Policy Evaluation
            while True:
                delta = 0
                for state in self.states:
                    v = value_function[state]
                    new_v = 0
                    for action, action_prob in enumerate(self.policy[state]):
                        for prob, next_state, reward, done in self.transition_probs[state][action]:
                            new_v += action_prob * prob * (reward + discount_factor * value_function[next_state] * (1 - done))
                    value_function[state] = new_v
                    delta = max(delta, abs(v - value_function[state]))
                if delta < theta:
                    break

            # Policy Improvement
            policy_stable = True
            for state in self.states:
                old_action_probs = np.copy(self.policy[state])
                action_values = np.zeros(len(self.actions))
                for action in self.actions:
                    for prob, next_state, reward, done in self.transition_probs[state][action]:
                        action_values[action] += prob * (reward + discount_factor * value_function[next_state] * (1 - done))
                best_action = np.argmax(action_values)
                new_policy = np.eye(len(self.actions))[best_action]
                self.policy[state] = new_policy
                if not np.array_equal(old_action_probs, self.policy[state]):
                    policy_stable = False
            if policy_stable:
                break

# Функція `task` тепер приймає `mode`
def task(env_name, mode='deterministic'):
    env = gym.make(env_name)
    transition_probability = env.unwrapped.P
    states = np.arange(env.unwrapped.observation_space.n)
    actions = [UP, RIGHT, DOWN, LEFT]
    
    # Передаємо обраний режим у клас
    policy_iteration = PolicyIteration(
        transition_probs=transition_probability,
        states=states,
        actions=actions,
        mode=mode
    )
    policy_iteration.run()
        
    rewards = []
    print(f"Тестуємо агента в режимі: '{mode}'...")
    for _ in tqdm.tqdm(range(500)):
        reward = play_episode(env, policy_iteration)
        rewards.append(reward)
    print(f"Середня винагорода: {np.mean(rewards):.3f} (std={np.std(rewards):.3f})")

    print("\nЗапускаємо візуалізацію одного епізоду...")
    env = gym.make(env_name, render_mode='human')
    reward = play_episode(env, policy_iteration, render=True, iter_max=100)
    print(f"Винагорода в цьому епізоді: {reward}")


if __name__ == "__main__":
    print("--- Завдання 5.1: Frozen Lake ---")
    
    # Додано інтерактивний вибір
    choice = ''
    while choice not in ['1', '2']:
        print("\nЯк має діяти агент у Frozen Lake?")
        print("1: Детерміністично (завжди обирати найкращий хід).")
        print("2: Стохастично (з крихтою випадковості, для дослідження 'слизької' поверхні).")
        choice = input("Ваш вибір (1 або 2): ")

    frozen_lake_mode = 'deterministic' if choice == '1' else 'stochastic'
    
    task('FrozenLake-v1', mode=frozen_lake_mode)

    print("\n\n--- Завдання 5.2: Cliff Walking ---")
    # Для Cliff Walking детермінований режим є оптимальним, оскільки середовище передбачуване
    print("Для цього завдання буде використано детерміністичний режим.")
    task('CliffWalking-v1', mode='deterministic')