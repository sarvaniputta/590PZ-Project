# Flood It

The game's objective is to turn the grid the same colour in as few flood fills as possible. 
By clicking on a colored cell, the color of the block is used to flood fill cells starting from 
top left (configurable in the code).
A prescribed number of moves are specified within which the player has to fill the whole grid.


## Requirements
- python=3.7
- numpy
- matplotlib
- tkinter

## How to run
Make changes in the flood.py game to select board size, number of colors and other parameters.
Then run

``python flood.py`` 

A tkinter window should pop up.
To make a move, you can click on any cell within the grid with the required color.

## Details
Every game starts with a random board generated.
Then depending on the board size, 
an optimal A* search or a heuristic search with limited depth is performed to obtain the solution.
The length of the solution is then presented as the move limit within which the board must be filled.
As a variation, the number of allowed moves for a single color is also specified.
The game still runs when the user exceeds the moves, 
with the moves being highlighted in red.
The user can see the actual solution using the show solution button.

## Caveats
It is hard to verify that there is only a unique solution for every problem. 
The current implementation does not use any transposition tables or any constraints to ensure this.
But, every solution within the prescribed moves is valid.
At larger board sizes, chances are there are optimal solutions which are strictly lower in moves
than the prescribed move number.



 


