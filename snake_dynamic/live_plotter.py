# In snake_dynamic/live_plotter.py
import matplotlib.pyplot as plt
import time

class LivePlotter:
    def __init__(self, title='Training...', xlabel='Number of Games', score_ylabel='Score', loss_ylabel='Loss'):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.suptitle(title)

        # Subplot 1: Scores
        self.ax1.set_xlabel(xlabel)
        self.ax1.set_ylabel(score_ylabel)
        self.line_score, = self.ax1.plot([], [], 'b-', label='Score')
        self.line_mean_score, = self.ax1.plot([], [], 'g-', label='Mean Score')
        self.ax1.legend(loc='upper left')
        self.ax1.grid()

        # Subplot 2: Losses
        self.ax2.set_xlabel(xlabel)
        self.ax2.set_ylabel(loss_ylabel)
        self.line_loss, = self.ax2.plot([], [], 'r-', label='Loss')
        self.line_mean_loss, = self.ax2.plot([], [], 'y-', label='Mean Loss')
        self.ax2.legend(loc='upper left')
        self.ax2.grid()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)
        plt.pause(0.1)

        self.data = {
            'scores': [], 'mean_scores': [],
            'losses': [], 'mean_losses': []
        }

    def update(self, new_data):
        self.data['scores'] = new_data.get('scores', self.data['scores'])
        self.data['mean_scores'] = new_data.get('mean_scores', self.data['mean_scores'])
        self.data['losses'] = new_data.get('losses', self.data['losses'])
        self.data['mean_losses'] = new_data.get('mean_losses', self.data['mean_losses'])
        
        # Update score plot
        self.line_score.set_data(range(len(self.data['scores'])), self.data['scores'])
        self.line_mean_score.set_data(range(len(self.data['mean_scores'])), self.data['mean_scores'])
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Update loss plot
        self.line_loss.set_data(range(len(self.data['losses'])), self.data['losses'])
        self.line_mean_loss.set_data(range(len(self.data['mean_losses'])), self.data['mean_losses'])
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.01) # Small sleep to allow GUI to process

def plot_process_target(queue):
    """
    This function is the target for the plotting process.
    It waits for data from the queue and updates the plot.
    """
    plotter = LivePlotter()
    while True:
        try:
            data = queue.get()
            if data is None: # Sentinel value to stop
                break
            plotter.update(data)
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            print(f"Plotting Error: {e}")
            break
    
    print("Plotting process finished.")
    # Keep window open until manually closed
    plt.show(block=True)