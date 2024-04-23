import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.optimize import linprog


class GameTheoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Theory Solver")

        # Initialize variables
        self.type_of_game_var = tk.StringVar(value="ZeroSum")
        self.num_moves_p1_var = tk.IntVar(value=2)
        self.num_moves_p2_var = tk.IntVar(value=2)
        self.payoffs = []

        # Create GUI widgets
        tk.Label(root, text="Type of Game:").grid(row=0, column=0, padx=10, pady=5)
        tk.OptionMenu(root, self.type_of_game_var, "ZeroSum", "NonZeroSum").grid(row=0, column=1, padx=10, pady=5)

        tk.Label(root, text="Number of moves for Player 1:").grid(row=1, column=0, padx=10, pady=5)
        tk.Entry(root, textvariable=self.num_moves_p1_var).grid(row=1, column=1, padx=10, pady=5)

        tk.Label(root, text="Number of moves for Player 2:").grid(row=2, column=0, padx=10, pady=5)
        tk.Entry(root, textvariable=self.num_moves_p2_var).grid(row=2, column=1, padx=10, pady=5)

        tk.Button(root, text="Enter Payoffs", command=self.enter_payoffs).grid(row=3, column=0, columnspan=2, padx=10,
                                                                               pady=10)

    def enter_payoffs(self):
        num_moves_p1 = self.num_moves_p1_var.get()
        num_moves_p2 = self.num_moves_p2_var.get()

        # Create a new window to enter payoffs
        payoff_window = tk.Toplevel(self.root)
        payoff_window.title("Enter Payoffs")

        self.payoff_entries = []

        # Labels for column numbers
        for j in range(num_moves_p2):
            col_label = tk.Label(payoff_window, text=f"player 2 move {j + 1}")
            col_label.grid(row=0, column=j + 1, padx=5, pady=5)

        # Labels for row numbers and payoff entries
        for i in range(num_moves_p1):
            row_label = tk.Label(payoff_window, text=f"player 1 move {i + 1}")
            row_label.grid(row=i + 1, column=0, padx=5, pady=5)
            row_entries = []
            for j in range(num_moves_p2):
                var = tk.StringVar(value="1,1")  # Default value for entry (Player 1 payoff, Player 2 payoff)
                entry = tk.Entry(payoff_window, textvariable=var, width=10)
                entry.grid(row=i + 1, column=j + 1, padx=5, pady=5)
                row_entries.append(var)
            self.payoff_entries.append(row_entries)

        # Button to calculate strategies
        tk.Button(payoff_window, text="Maximin moves", command=self.Maximin_move).grid(row=num_moves_p1 + 2, columnspan=num_moves_p2, padx=10, pady=10)
        tk.Button(payoff_window, text="Iterated Elimination of weakly Dominated", command=self.Weak_Dominance_Elimination).grid(row=num_moves_p1 + 3, columnspan=num_moves_p2, padx=10, pady=10)
        tk.Button(payoff_window, text="Iterated Elimination of Strictly Dominated", command=self.Strict_Dominance_Elimination).grid(row=num_moves_p1 + 4, columnspan=num_moves_p2, padx=10, pady=10)
        tk.Button(payoff_window, text="Best responses", command=self.Best_responses).grid(row=num_moves_p1 + 5, columnspan=num_moves_p2, padx=10, pady=10)
        tk.Button(payoff_window, text="Nash equilibria", command=self.Nash_equilibria).grid(row=num_moves_p1 + 6, columnspan=num_moves_p2, padx=10, pady=10)
        tk.Button(payoff_window, text="Mixed strategys", command=self.Mixed_strategys).grid(row=num_moves_p1 + 7, columnspan=num_moves_p2, padx=10, pady=10)



        
    
    def Maximin_move(self):
        self.ZeroSum_check() # check ZeroSum
        # Convert the payoffs to a numpy array
        payoffs_p1 = np.array([[float(entry.get().split(',')[0]) for entry in row] for row in self.payoff_entries])
        payoffs_p2 = np.array([[float(entry.get().split(',')[1]) for entry in row] for row in self.payoff_entries])

        # Min max for player 1
        min_payoffs_p1 = np.min(payoffs_p1, axis=1)
        maximin_move_p1 = np.argmax(min_payoffs_p1)

        # Min max for player 2
        min_payoffs_p2 = np.min(payoffs_p2, axis=0)
        maximin_move_p2 = np.argmax(min_payoffs_p2)

        messagebox.showinfo("Maximin moves", f"The maximin move for player 1 is move {maximin_move_p1 + 1}, and for player 2 is move {maximin_move_p2 + 1}")

    def Strict_Dominance_Elimination(self):

        self.ZeroSum_check() # check ZeroSum
        # Convert the payoffs to a numpy array
        payoffs = np.array([[entry.get().split(',') for entry in row] for row in self.payoff_entries], dtype=float)

        eliminated_version = payoffs

        # Loop for elimination
        while True:
            # Find strictly dominated strategies for player 1
            dominated_strategies_p1 = []
            for i in range(eliminated_version.shape[0]):
                for j in range(eliminated_version.shape[0]):
                    if i != j and all(eliminated_version[i, :, 0] < eliminated_version[j, :, 0]):
                        dominated_strategies_p1.append(i)
                        break

            # Find strictly dominated strategies for player 2
            dominated_strategies_p2 = []
            for i in range(eliminated_version.shape[1]):
                for j in range(eliminated_version.shape[1]):
                    if i != j and all(eliminated_version[:, i, 1] < eliminated_version[:, j, 1]):
                        dominated_strategies_p2.append(i)
                        break
            if not dominated_strategies_p1 and not dominated_strategies_p2:
                break

            # Remove the strictly dominated strategies
            eliminated_version = np.delete(eliminated_version, dominated_strategies_p1, axis=0)
            eliminated_version = np.delete(eliminated_version, dominated_strategies_p2, axis=1)

        messagebox.showinfo("Eliminated version by strict dominance",
                            f"The eliminated version of the game by strict dominance is:\n{eliminated_version}")

    def Weak_Dominance_Elimination(self):

        self.ZeroSum_check() # check ZeroSum
        # Convert the payoffs to a numpy array
        payoffs = np.array([[entry.get().split(',') for entry in row] for row in self.payoff_entries], dtype=float)
        eliminated_version = payoffs

        # Loop for elimination
        while True:
            # Find weakly dominated strategies for player 1
            dominated_strategies_p1 = []
            for i in range(eliminated_version.shape[0]):
                for j in range(eliminated_version.shape[0]):
                    if i != j and all(eliminated_version[i, :, 0] <= eliminated_version[j, :, 0]) and any(
                            eliminated_version[i, :, 0] < eliminated_version[j, :, 0]):
                        dominated_strategies_p1.append(i)
                        break

            # Find weakly dominated strategies for player 2
            dominated_strategies_p2 = []
            for i in range(eliminated_version.shape[1]):
                for j in range(eliminated_version.shape[1]):
                    if i != j and all(eliminated_version[:, i, 1] <= eliminated_version[:, j, 1]) and any(
                            eliminated_version[:, i, 1] < eliminated_version[:, j, 1]):
                        dominated_strategies_p2.append(i)
                        break
            if not dominated_strategies_p1 and not dominated_strategies_p2:
                break

            # Remove the weakly dominated strategies
            eliminated_version = np.delete(eliminated_version, dominated_strategies_p1, axis=0)
            eliminated_version = np.delete(eliminated_version, dominated_strategies_p2, axis=1)

        messagebox.showinfo("Eliminated version by weak dominance",
                            f"The eliminated version of the game by weak dominance is:\n{eliminated_version}")

    def Best_responses(self):

        self.ZeroSum_check() # check ZeroSum
        # Convert the payoffs to a numpy array
        payoffs = np.array([[entry.get().split(',') for entry in row] for row in self.payoff_entries], dtype=float)
        best_responses_p1 = []
        best_responses_p2 = []

        # Find the best responses for player 1
        for j in range(payoffs.shape[1]):
            max_payoff_p1 = np.max(payoffs[:, j, 0])
            best_responses_p1.append([i + 1 for i in range(payoffs.shape[0]) if payoffs[i, j, 0] == max_payoff_p1])

        # Find the best responses for player 2
        for i in range(payoffs.shape[0]):
            max_payoff_p2 = np.max(payoffs[i, :, 1])
            best_responses_p2.append([j + 1 for j in range(payoffs.shape[1]) if payoffs[i, j, 1] == max_payoff_p2])

        # Format the best responses
        best_responses_p1_str = ', '.join(
            f"\nplayer 2 move {j + 1} -----> player 1 move {i}" for j, moves in enumerate(best_responses_p1) for i in
            moves)
        best_responses_p2_str = ', '.join(
            f"\nplayer 1 move {i + 1} -----> player 2 move {j}" for i, moves in enumerate(best_responses_p2) for j in
            moves)

        messagebox.showinfo("Best responses",
                            f"Best responses for player 1: {best_responses_p1_str}\n\nBest responses for player 2: {best_responses_p2_str}")

    def Nash_equilibria(self):

        self.ZeroSum_check() # check ZeroSum
        # Convert the payoffs to a numpy array
        payoffs = np.array([[entry.get().split(',') for entry in row] for row in self.payoff_entries], dtype=float)
        nash_equilibria = []

        # Find nash equilibria
        for i in range(payoffs.shape[0]):
            for j in range(payoffs.shape[1]):
                # If a strategy is a best response to the other player's strategy, it is a Nash equilibrium
                if all(payoffs[:, j, 0] <= payoffs[i, j, 0]) and all(payoffs[i, :, 1] <= payoffs[i, j, 1]):
                    nash_equilibria.append((i + 1, j + 1))

        # Format the Nash equilibria
        nash_equilibria_str = ', '.join(f"\nplayer 1 move {i} <-----> player 2 move {j}" for i, j in nash_equilibria)

        messagebox.showinfo("Nash equilibria", f"Nash equilibria: {nash_equilibria_str}")



    def Mixed_strategys(self):
        return None

 
    # *******************************************************************************************************************************
    # *******************************************************************************************************************************
    # *******************************************************************************************************************************
   
    def ZeroSum_check(self):

        game_type = self.type_of_game_var.get()
        payoffs_p1 = np.array([[float(entry.get().split(',')[0]) for entry in row] for row in self.payoff_entries])
        payoffs_p2 = np.array([[float(entry.get().split(',')[1]) for entry in row] for row in self.payoff_entries])

        if game_type == "ZeroSum":
            if np.array_equal(payoffs_p1 + payoffs_p2, np.zeros_like(payoffs_p1)):
                return
            else:
                messagebox.showerror("Error", "Invalid input format.  it not Zero Sum")
                quit()
                
        else:        
            return

if __name__ == "__main__":
    root = tk.Tk()
    app = GameTheoryApp(root)
    root.mainloop()




