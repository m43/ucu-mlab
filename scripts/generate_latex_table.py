csv_path = "/home/user72/Desktop/epfl-vimam -- Results of robustness analysis - Pivot Table 1.csv"

with open(csv_path) as f:
    lines = f.read().split("\n")

print("\\band")
for line in lines[4:-1]:
    line = line.replace("_", " ")
    parts = line.split(",")
    if parts[0] == "midrule":
        print("\\midrule")
        continue

    x = f"\\texttt{{{parts[0]}}}"
    x += f" & {parts[1].split(']')[1]}"
    x += f" & " + ('\\checkmark' if 'V' in parts[1].split(']')[0] else '')
    x += f" & " + ('\\checkmark' if 'D' in parts[1].split(']')[0] else '')
    for i in range(6):
        if i == 3:
            continue
        x += f" && {float(parts[2 + i * 3 + 0].replace('%', '')):.2f}"
        x += f" & {float(parts[2 + i * 3 + 1].replace('%', '')):.2f}"
        x += f" & {float(parts[2 + i * 3 + 2].replace('%', '')):.2f}"
    x += " \\\\"

    print(x)
print("\\midrule")