import pandas as pd 
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

# خواندن داده‌ها
df1 = pd.read_excel("C:/Users/this pc/Desktop/tamrin1_behine/houses.xls")
df2 = pd.read_excel("C:/Users/this pc/Desktop/tamrin1_behine/schools.xls")

X_houses = df1['POINT_X'].to_numpy()
Y_houses = df1['POINT_Y'].to_numpy()
X_schools = df2['POINT_X'].to_numpy()
Y_schools = df2['POINT_Y'].to_numpy()

# تنظیمات اولیه
population_size = 80
num_houses = 380 
num_schools = 10 
max_houses_per_school = 38 
mutation_rate = 0.1      
crossover_rate = 0.8     

# تولید یک فرد بهینه
def generate_individual():
    individual = [i for i in range(1, num_schools + 1) for _ in range(max_houses_per_school)]
    random.shuffle(individual)
    return individual

population = [generate_individual() for _ in range(population_size)]

population1=population.copy()
# مرحله ۱: محاسبه‌ی فاصله‌ی اولیه بین هر خانه و مدرسه و ذخیره در یک ماتریس
distance_matrix = np.zeros((num_houses, num_schools))
for i in range(num_houses):
    for j in range(num_schools):
        dx = X_houses[i] - X_schools[j]
        dy = Y_houses[i] - Y_schools[j]
        distance_matrix[i][j] = dx**2 + dy**2 # فاصله بدون ریشه برای سرعت بیشتر

# مرحله ۲: بازنویسی تابع fitness بر اساس distance_matrix
def fitness(individual):
    indices = np.array(individual) - 1 # چون شماره مدارس از 1 شروع می‌شن
    house_indices = np.arange(num_houses)
    return distance_matrix[house_indices, indices].sum()

# انتخاب والدین با چرخ رولت سریع‌تر
def roulette_wheel_selection(population):
    fitness_values = np.array([fitness(ind) for ind in population])
    inv_fit = 1 / fitness_values
    probs = inv_fit / inv_fit.sum()
    selected = np.random.choice(len(population), size=2, replace=False, p=probs)
    return population[selected[0]], population[selected[1]]

# اصلاح کودک در صورت داشتن خانه‌های اضافی
def fix_child(child):
    count = Counter(child)
    for school, c in count.items():
        while c > max_houses_per_school:
            idxs = [i for i, val in enumerate(child) if val == school]
            for i in idxs:
                available = [s for s in range(1, num_schools + 1) if child.count(s) < max_houses_per_school]
                if available:
                    new_school = random.choice(available)
                    child[i] = new_school
                    c -= 1
                    if c <= max_houses_per_school:
                        break
    return child

# ترکیب والدین و اصلاح فرزندان
def crossover(parent1, parent2):
    point = random.randint(1, num_houses - 1)
    child1 = fix_child(parent1[:point] + parent2[point:])
    child2 = fix_child(parent2[:point] + parent1[point:])
    return child1, child2

# جهش ساده
def mutation(individual):
    i, j = random.sample(range(num_houses), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual

# الگوریتم ژنتیک
generations = 10000
best_solutions = []
best_fits = []
generation1=[]

elitism_count = 15 # تعداد نخبه‌هایی که به نسل بعد منتقل می‌کنیم

for generation in range(generations):
    # مرتب‌سازی جمعیت فعلی بر اساس فیتنس (از کم به زیاد)
    sorted_population = sorted(population, key=fitness)
    elites = sorted_population[:elitism_count] # انتخاب بهترین‌ها

    new_population = elites.copy() # شروع جمعیت جدید با افراد نخبه

    while len(new_population) < population_size:
        if random.random() < crossover_rate:
            parent1, parent2 = roulette_wheel_selection(population)
            offspring1, offspring2 = crossover(parent1, parent2)
        else:
            # بدون crossover، والدین مستقیماً کپی می‌شن
            offspring1, offspring2 = random.sample(population, 2)
            offspring1 = offspring1.copy()
            offspring2 = offspring2.copy()
            
            # اعمال جهش با احتمال مشخص
        if random.random() < mutation_rate:
            offspring1 = mutation(offspring1)
        if random.random() < mutation_rate:
            offspring2 = mutation(offspring2)

        new_population.extend([offspring1, offspring2])

    population = new_population[:population_size] # اگه بیشتر شد، اضافه‌ها رو حذف کن

    best_solution = min(population, key=fitness)
    best_fit = fitness(best_solution)
    best_solutions.append(best_solution)
    best_fits.append(best_fit)
    generation1.append(generation)
    print(f"Generation {generation}: Fitness = {best_fits[-1]:.2f}")
# پیدا کردن بهترین جواب
best_solution = min(best_solutions, key=fitness)
best_fit = fitness(best_solution)
print(f"\nBest Total Fitness: {best_fit:.2f}")

# --- گرفتن بهترین جواب از نسل اول ---
first_gen_best = min(population1, key=fitness)

# --- گرفتن بهترین جواب از کل نسل‌ها (قبلاً محاسبه شده) ---
final_best = best_solution  # از قبل در انتهای الگوریتم ذخیره شده

# --- تابع برای شمارش تعداد خانه‌های تخصیص‌یافته به هر مدرسه ---
def count_assignments(solution):
    counts = Counter(solution)
    for school_id in range(1, num_schools + 1):
        assigned = counts.get(school_id, 0)
        print(f"School {school_id}: {assigned} houses assigned {'✅' if assigned <= max_houses_per_school else '❌'}")
    return counts

# --- تابع برای رسم نمودار تخصیص ---
def plot_comparison(first, final):
    plt.figure(figsize=(14, 6))

    # --- اولی: نسل اول ---
    plt.subplot(1, 2, 1)
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
    for i in range(len(first)):
        sx, sy = X_schools[first[i] - 1], Y_schools[first[i] - 1]
        hx, hy = X_houses[i], Y_houses[i]
        color = colors[first[i] - 1]
        plt.plot([sx, hx], [sy, hy], color=color, linewidth=0.5)
        plt.scatter(hx, hy, color=color, s=8)
    plt.scatter(X_schools, Y_schools, color='red', marker='*', s=200, label='Schools')
    plt.title("Generation 1 - Initial Solution")
    plt.grid(True)

    # --- دومی: جواب نهایی ---
    plt.subplot(1, 2, 2)
    for i in range(len(final)):
        sx, sy = X_schools[final[i] - 1], Y_schools[final[i] - 1]
        hx, hy = X_houses[i], Y_houses[i]
        color = colors[final[i] - 1]
        plt.plot([sx, hx], [sy, hy], color=color, linewidth=0.5)
        plt.scatter(hx, hy, color=color, s=8)
    plt.scatter(X_schools, Y_schools, color='red', marker='*', s=200, label='Schools')
    plt.title("Best Solution - Final Generation")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --- نمایش نمودارها ---
plot_comparison(first_gen_best, final_best)

# --- شمارش و بررسی تخصیص نهایی ---
print("\n📊 Final Assignment Summary:")
assignment_counts = count_assignments(final_best)
