import numpy as np
import matplotlib.pyplot as plt

def visualize(full_df,
              model,
              feature_cols=None,
              coord_cols=('row', 'col'),
              target_col='SD',
              aoi_name_col=None,
              aoi_name=None,
              cmap='viridis',
              s=5,
              figsize=(12, 6)):
    """
    Plot actual vs predicted snow depth maps from a full DataFrame.
    """
    df = full_df
    # Filter by aoi if requested
    if aoi_name_col and aoi_name is not None:
        df = df[df[aoi_name_col] == aoi_name]

    # Extract coordinates (assuming coord_cols = (Y_column, X_column))
    ys = df[coord_cols[0]].values
    xs = df[coord_cols[1]].values

    # Determine feature columns
    if feature_cols is None:
        exclude = set(coord_cols) | {target_col}
        if aoi_name_col:
            exclude.add(aoi_name_col)
        feature_cols = [col for col in df.columns if col not in exclude]

    # Prepare data for prediction
    X = df[feature_cols].values
    y_true = df[target_col].values

    # Predict
    y_pred = model.predict(X)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Actual
    sc0 = axes[0].scatter(xs, ys, c=y_true, s=s, cmap=cmap)
    axes[0].set_title('Actual Snow Depth')
    axes[0].set_xlabel(coord_cols[1])
    axes[0].set_ylabel(coord_cols[0])
    plt.colorbar(sc0, ax=axes[0], label=target_col)

    # Predicted
    sc1 = axes[1].scatter(xs, ys, c=y_pred, s=s, cmap=cmap)
    axes[1].set_title('Predicted Snow Depth')
    axes[1].set_xlabel(coord_cols[1])
    axes[1].set_ylabel(coord_cols[0])
    plt.colorbar(sc1, ax=axes[1], label=target_col)

    plt.tight_layout()
    plt.show()

