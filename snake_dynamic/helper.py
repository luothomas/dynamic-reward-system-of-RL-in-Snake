import matplotlib.pyplot as plt
from IPython import display

plt.ion()

# 新增 losses 和 mean_losses 參數
def plot(scores, mean_scores, losses=None, mean_losses=None): # plotting
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    # 判斷是否要繪製 Loss (如果傳入了 losses 參數)
    num_subplots = 1
    if losses is not None and mean_losses is not None:
        num_subplots = 2

    # Subplot 1: Scores
    plt.subplot(num_subplots, 1, 1)
    plt.title('Training...')
    if num_subplots == 1: # 如果只有一個子圖，則顯示 X 軸標籤
        plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend(loc='upper left')

    # Subplot 2: Losses (如果提供了 losses 數據)
    if num_subplots == 2:
        plt.subplot(num_subplots, 1, 2)
        # plt.title('Training Loss...') # 子標題可選
        plt.xlabel('Number of Games')
        plt.ylabel('Loss')
        plt.plot(losses, label='Loss', color='red')
        plt.plot(mean_losses, label='Mean Loss', color='orange')
        if losses:
            plt.text(len(losses)-1, losses[-1], f"{losses[-1]:.4f}")
        if mean_losses:
            plt.text(len(mean_losses)-1, mean_losses[-1], f"{mean_losses[-1]:.4f}")
        plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)