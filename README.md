# Texas-Hold-Em
(WIP) Computes the psychology-neutral odds of winning a hand at Texas Hold'Em.

Each card is assigned a unique ID from 0 to 51 (for translation to values that mean something to human players, refer to the SORTES and RANGS vectors and the indices_tbl dataframe).

Feed the algorithm an array containing the player's cards' indices, an array containing any known table cards, and a scalar indicating the number of opponents, and it will return the psychology-neutral probability of winning the hand.

Not quite done testing so the algorithm will still give off some errors and might still break in certain scenarios; it also has the potential of getting very heavy, as the number of known table cards tends to zero and the number of opponents grows.
