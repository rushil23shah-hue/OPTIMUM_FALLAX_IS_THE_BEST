when : 5/7 

what : Exploring Bipedal Walker environment

why : To study how specification gaming and reward hacking occurs in different environments 

how : in explore_env.py understood the observation space(24 values),action space(4 values ranging from -1 to 1 ) , understood various action type random,manual and algo based.Understood roles of learn() and predict() functions. 

---

when : 6/7 

what : implemention of PPO baseline 

why : task 1 was just to make the model cross the line and training an baseline agent . No tuning of hyperparameters . one leg drag and hop motion , Reward in the range of ~280 

how : Basic code structure format , changing the total_timesteps (from 10k to 1 M ) , finally converged at 1M . 


---

when : 7/7 

what : Learned about on and off policy , SAC structure understood and implemented . Converged at 500K steps and much more natural walking style than PPO 

why : to compare with PPO ans check which is a better algo 

how : again as just baseline training was done no code changes just replaced PPO with SAC. 

---
when : 8/7 

what : git push 

why : documentation 

how : git init , git clone ,git branch , git checkout Pacchu, git add . , git commit -m "Intial Commit PPO and SAC " , git push origin Pacchu 

---


https://docs.google.com/document/d/1EExbpN5V9y3Q-Ix-Z3Al1N3gKadD6E8mL-FX0Px7avs/edit?pli=1&tab=t.0

REFERENCES : https://www.youtube.com/watch?v=ApG0lWv6gGc

https://www.youtube.com/watch?v=ApG0lWv6gGc

https://www.youtube.com/watch?v=aGOyF5aU9zc

https://youtube.com/playlist?list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte&si=avz2-GRcsObuXT1h
