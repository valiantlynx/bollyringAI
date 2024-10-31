# BollyringAI
ai algorithm to best assign calls to the best worker

## round1 -fight
Call schedule by wednesday 30.10.2024 23:59

things we have to go by:
 - [x] call_time vs call_profit
   - seems like the calls with the most recommendation are the ones that are take less than 30 min
 - [x] likely_to_recommend
 - [x] some workers like: w_f5400776-c6cc-4563-b835-b5ee56242aa1 are paid a lot and very high likely_to_recommend
 - [ ] we have 755 workers
 - [ ] Mail Sent about it: it seems the call_profit is maybe calculated wrong. for example a worker sold something for 695, the company paid a commmision for 834.0 and the call profit says its 834 for the same sale
  
 - commision persentages
   - hard: 120%
   - medium: 100%
   - easy: 80%
 - how about cutting the commision

### how to make a schedule
 - give more calls to the better workers
 - give hard calls to the better workers: 
   - threshold in likely to recommend average
   - >2.2 hard calls
   - 1.8-2.2 medium calls
   - <1.8 easy calls
 - 
 - if we can change commision
   - for example just cut the hard commision in half so the workers are not motivated to take them 
   - use hard, medium and easy
 - else 
   - use just easy
  
we got our results reports and format structure