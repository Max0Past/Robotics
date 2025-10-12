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
    def __init__(self, transition_probs, states, actions):
        self.transition_probs = transition_probs
        self.states = states
        self.actions = actions
        # Початкова стратегія залишається рівномірною
        self.policy = np.ones([len(self.states), len(self.actions)]) / len(self.actions)

    def pick_action(self, obs):
        ### Using a policy pick an action for the state `obs`
        ### Використовуючи стратегію обрати дію для стану `obs`
        
        # --- ЗМІНЕНО ТУТ ---
        # Замість того, щоб завжди вибирати найкращу дію (argmax),
        # ми вибираємо дію випадково, відповідно до ймовірностей у нашій стратегії.
        # Це дозволяє агенту досліджувати середовище.
        return np.random.choice(self.actions, p=self.policy[obs])


    def run(self):
        ### Using `self.transition_probs`, `self.states`, and `self.actions`, compute a policy.
        ### Викорстовуючи `self.transition_probs`, `self.states` та `self.actions`, обчисліть стратегію.
        
        ### Зверніть увагу: | Note:
        ### [(prob, next_state, reward, terminate), ...] = transition_probability[state][action]
        ### prob = probability(next_state | state, action)
        
        # Гіперпараметри
        theta = 1e-9  # Зробимо поріг трохи меншим для кращої точності
        discount_factor = 0.99  # Коефіцієнт дисконтування для майбутніх винагород

        # 1. Ініціалізація
        value_function = np.zeros(len(self.states))

        # Починаємо основний цикл алгоритму
        while True:
            # -- Крок 2: Оцінка стратегії (Policy Evaluation) --
            while True:
                delta = 0
                for state in self.states:
                    v = value_function[state]
                    new_v = 0
                    
                    for action, action_prob in enumerate(self.policy[state]):
                        for prob, next_state, reward, done in self.transition_probs[state][action]:
                            # --- НЕВЕЛИКЕ ВИПРАВЛЕННЯ ТУТ ---
                            # Додаємо `* (1-done)`, щоб цінність майбутніх станів не враховувалася, 
                            # якщо гра закінчилася (агент впав в ополонку або дійшов до мети).
                            new_v += action_prob * prob * (reward + discount_factor * value_function[next_state] * (1 - done))
                    
                    value_function[state] = new_v
                    delta = max(delta, abs(v - value_function[state]))

                if delta < theta:
                    break

            # -- Крок 3: Поліпшення стратегії (Policy Improvement) --
            policy_stable = True
            for state in self.states:
                old_action_probs = np.copy(self.policy[state])

                action_values = np.zeros(len(self.actions))
                for action in self.actions:
                    for prob, next_state, reward, done in self.transition_probs[state][action]:
                         # --- І ТУТ ТАКЕ САМЕ ВИПРАВЛЕННЯ ---
                        action_values[action] += prob * (reward + discount_factor * value_function[next_state] * (1 - done))
                
                best_action = np.argmax(action_values)
                
                # Оновлюємо стратегію, роблячи її детермінованою (жадібною)
                # на етапі поліпшення.
                new_policy = np.eye(len(self.actions))[best_action]
                self.policy[state] = new_policy
                
                # Перевіряємо, чи змінилася стратегія
                if not np.array_equal(old_action_probs, self.policy[state]):
                    policy_stable = False
            
            if policy_stable:
                break

# --- РЕШТА КОДУ ЗАЛИШАЄТЬСЯ БЕЗ ЗМІН ---
def task(env_name):
    # Для FrozenLake краще використовувати неслизьку версію для налагодження,
    # але для демонстрації стохастичності залишимо стандартну.
    # env = gym.make(env_name, is_slippery=False)
    env = gym.make(env_name)
    transition_probability = env.unwrapped.P
    states = np.arange(env.unwrapped.observation_space.n)
    actions = [UP, RIGHT, DOWN, LEFT]
    policy_iteration = PolicyIteration(
        transition_probs=transition_probability,
        states=states,
        actions=actions
    )
    policy_iteration.run()
        
    rewards = []
    # Запустимо більше епізодів для надійнішої статистики
    for _ in tqdm.tqdm(range(500)):
        reward = play_episode(env, policy_iteration)
        rewards.append(reward)
    print(f"Average reward: {np.mean(rewards):.3f} (std={np.std(rewards):.3f})")

    env = gym.make(env_name, render_mode='human')
    reward = play_episode(env, policy_iteration, render=True, iter_max=100)


if __name__ == "__main__":
    print("Task 5.1 - Frozen Lake")
    task('FrozenLake-v1')

    print("Task 5.2 - Cliff Walking")
    task('CliffWalking-v1')