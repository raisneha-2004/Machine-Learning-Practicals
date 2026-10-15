num_classes = 5
epsilon = 0.1

labels = [1, 0, 0, 0, 0]
smoothed_labels = []

for label in labels:
    new_label = (1 - epsilon) * label + epsilon / num_classes
    smoothed_labels.append(round(new_label, 2))

print("Original Labels:", labels)
print("Smoothed Labels:", smoothed_labels)
print("Total Probability:", sum(smoothed_labels))