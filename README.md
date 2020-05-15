# Texas-Hold-Em
Computes the psychology-neutral odds of winning a given hand at Texas Hold'Em poker.

Each card is assigned a unique ID from 0 to 51 (for translation to values that mean something to human players, refer to the SORTES and RANGS vectors and the indices_tbl dataframe).

Feed the algorithm a vector containing the indices of the player's cards, a second vector containing the indices of any known table cards, and a scalar indicating the number of opponents, and it will return the psychology-neutral probability of winning the hand.

The algorithm gets very heavy if less than 4 table cards are known or if there are more than 2 opponents, in which case the simulation function will give much quicker, albeit less precise, results (although you can ultimately chose your preferred tradeoff between speed and precision by adjusting the ITER parameter of the simulation function, perhaps using credibility theory).
