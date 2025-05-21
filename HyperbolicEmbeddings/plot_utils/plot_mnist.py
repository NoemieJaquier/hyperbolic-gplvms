import numpy as np

def plot_predictions(axs_predictions, mean_predictions, data_mean, data_std, fontsize, yoffset, apply_offset=True):
    n_rows, n_cols = 2, 5
    N_eval = mean_predictions.shape[0]
    prediction_indices = np.linspace(0, N_eval-1, n_rows*n_cols)
    t = np.linspace(0, 1, n_rows*n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            ax_prediction = axs_predictions[i, j]
            prediction_index = int(prediction_indices[i*n_cols + j])
            img = (mean_predictions[prediction_index] * data_std + data_mean).detach().numpy().reshape(28, 28)
            ax_prediction.set_title(f"t={t[i*n_cols + j]:.2f}", fontsize=fontsize, pad=-5)
            ax_prediction.imshow(img, cmap='gray')
            ax_prediction.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax_prediction.spines.values():
                spine.set_visible(False)
            if apply_offset:
                if i == 1:
                    ax_pos = ax_prediction.get_position()
                    ax_prediction.set_position([ax_pos.x0, ax_pos.y0 + 0.03, ax_pos.width, ax_pos.height])
                ax_pos = ax_prediction.get_position()
                ax_prediction.set_position([ax_pos.x0 - 0.2, ax_pos.y0 + yoffset, ax_pos.width, ax_pos.height])