import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Встановлюємо інтерактивний бекенд
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import argparse

# Ця програма виконує аналіз спектру з метою виявлення піків та отримання додаткових параметрів піків, використовуючи Python і бібліотеку Tkinter для графічного інтерфейсу.

parser = argparse.ArgumentParser()
parser.add_argument('-i', help="path to input .csv", default="output.csv")
args = parser.parse_args()

# 1. Завантаження Даних: Програма використовує файл .csv, що містить спектральні дані. Дані завантажуються за допомогою функції load_spectrum_from_file.
# Функція для завантаження спектру з файлу
def load_spectrum_from_file(path):
    try:
        data = np.loadtxt(path, delimiter=",")
        return data[:, 4]
    except Exception as e:
        messagebox.showerror("Помилка", f"Не вдалося зчитати спектр з файлу: {str(e)}")
        return None

# 2. Обробка Спектру: Після завантаження спектру виконується згладжування даних за допомогою функції smooth_spectrum, яка використовує фільтр Гаусса для усунення шумів та покращення якості спектру.
# Функція для згладжування спектру
def smooth_spectrum(spectrum, window_size=5):
    return gaussian_filter1d(spectrum, sigma=window_size)

# Гауссова функція для апроксимації піків
def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

# 3. Пошук Піків: Згладжений спектр обробляється для пошуку піків. Для цього використовується друга похідна згладженого спектру.
# Піки визначаються за допомогою функції find_peaks_in_spectrum, яка знаходить максимуми в другій похідній спектру.
# Також здійснюється апроксимація кожного піка гауссовою функцією для визначення додаткових параметрів, таких як ширина на половині висоти (ΔE), площа під піком та інші характеристики.
# Функція для пошуку піків у спектрі та розрахунку додаткових параметрів
def find_peaks_in_spectrum(smoothed_spectrum):
    # Обчислення другої похідної для виявлення піків
    second_derivative = np.gradient(np.gradient(smoothed_spectrum))
    peak_indices, _ = find_peaks(-second_derivative, height=0.0005)

    # Збір інформації про знайдені піки
    peaks_info = []
    for idx in peak_indices:
        amplitude = smoothed_spectrum[idx]

        # Визначення області навколо піка для апроксимації
        window_size = 10
        start = max(0, idx - window_size)
        end = min(len(smoothed_spectrum), idx + window_size)
        x_data = np.arange(start, end)
        y_data = smoothed_spectrum[start:end]

        # Апроксимація гауссовою функцією для визначення параметрів піка
        try:
            popt, _ = curve_fit(gaussian, x_data, y_data, p0=[amplitude, idx, 5])
            sigma_gaussian = abs(popt[2])
            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma_gaussian  # Ширина на половині висоти (ΔE)
            area = np.trapz(gaussian(x_data, *popt), x_data)  # Площа під піком
        except RuntimeError:
            sigma_gaussian = fwhm = area = np.nan  # Якщо апроксимація не вдалася, ставимо NaN

        # Визначення сігми методом найменших квадратів
        if not np.isnan(sigma_gaussian):
            #Рядок sigma_mnk = np.std(y_data) обчислює стандартне відхилення значень у вибраному вікні навколо піка (y_data).
            #  Стандартне відхилення (np.std()) є мірою розсіювання значень навколо середнього значення, і це дозволяє оцінити "ширину" піка в даних.
            #  У даному випадку значення sigma_mnk використовується як наближення для визначення ширини піка методом найменших квадратів.
            sigma_mnk = np.std(y_data) 
        else:
            sigma_mnk = np.nan

        peaks_info.append((idx, amplitude, area, sigma_gaussian, fwhm, sigma_mnk))

    return peaks_info, second_derivative

# 4. Візуалізація Результатів: Після обробки спектру результати відображаються на графіках.
# Графіки показують оригінальний спектр, згладжений спектр, а також виявлені піки.
# Додатково будується графік другої похідної, де також відображаються знайдені піки.
# Функція для побудови графіку спектру з піками
def plot_spectrum(spectrum, peaks_info, smoothed_spectrum):
    display_peak_info(peaks_info)
    
    # Вікно для спектру та піків
    plt.figure(figsize=(10, 5))
    plt.plot(spectrum, label='Original Spectrum')
    plt.plot(smoothed_spectrum, label='Smoothed Spectrum', linestyle='--')

    # Додавання піків на графік
    if peaks_info:
        peak_indices = [idx for idx, _, _, _, _, _ in peaks_info]
        peak_amplitudes = [amplitude for _, amplitude, _, _, _, _ in peaks_info]
        plt.plot(peak_indices, peak_amplitudes, 'ro', label='Peaks')

    plt.legend()
    plt.grid(True)
    plt.xlabel('Channel')
    plt.ylabel('Intensity')
    plt.title('Spectrum with Detected Peaks')
    plt.show()

# Функція для побудови графіку другої похідної з піками
# Додатково на графік додається графік першої похідної
def plot_second_derivative(second_derivative, peak_indices, first_derivative):
    # Вікно для другої похідної
    plt.figure(figsize=(10, 5))
    plt.plot(second_derivative, label='Second Derivative', color='g')
    plt.plot(first_derivative, label='First Derivative', color='b')

    # Додавання піків на графік другої похідної
    if peak_indices:
        plt.plot(peak_indices, second_derivative[peak_indices], 'ro', label='Peaks on Second Derivative')

    plt.legend()
    plt.grid(True)
    plt.xlabel('Channel')
    plt.ylabel('Derivative')
    plt.title('First and Second Derivatives of the Spectrum with Detected Peaks')
    plt.show()

# Функція для побудови графіку виключно другої похідної з піками
def plot_only_second_derivative(second_derivative, peak_indices):
    # Вікно для другої похідної
    plt.figure(figsize=(10, 5))
    plt.plot(second_derivative, label='Second Derivative', color='g')

    # Додавання піків на графік другої похідної
    if peak_indices:
        plt.plot(peak_indices, second_derivative[peak_indices], 'ro', label='Peaks on Second Derivative')

    plt.legend()
    plt.grid(True)
    plt.xlabel('Channel')
    plt.ylabel('Second Derivative')
    plt.title('Second Derivative of the Spectrum with Detected Peaks')
    plt.show()

# 5. Збереження Результатів: Інформація про піки (канал, амплітуда, площа під піком, ширина на половині висоти та інші параметри) зберігається у файл output_procesed.csv.
# Функція для збереження інформації про піки у файл
def save_peaks_info_to_file(peaks_info):
    try:
        with open("output_procesed.csv", 'w') as file:
            file.write("Канал,Амплітуда,Площа,σ_гаус,ΔE,σ_мнк\n")

            for idx, amplitude, area, sigma_gaussian, fwhm, sigma_mnk in peaks_info:
                file.write(f"{idx},{amplitude:.2f},{area:.2f},{sigma_gaussian:.2f},{fwhm:.2f},{sigma_mnk:.2f}\n")

        print("Інформацію про піки збережено у файл output_procesed.csv")
    except Exception as e:
        messagebox.showerror("Помилка", f"Не вдалося зберегти інформацію про піки: {str(e)}")

# 6. Графічний Інтерфейс: Програма має простий графічний інтерфейс користувача, який дозволяє запускати аналіз спектру, а також відображає результати у вигляді тексту та графіків.
# Функція для аналізу спектру
def analyze_spectrum():
    # Отримання шляху до файлу з аргументів командного рядка
    filename = args.i
    spectrum = load_spectrum_from_file(filename)
    if spectrum is not None:
        # Визначення window_size як 3% від кількості каналів у спектрі
        window_size = int(len(spectrum) * 0.03)
        smoothed_spectrum = smooth_spectrum(spectrum, window_size=window_size)
        first_derivative = np.gradient(smoothed_spectrum)  # Обчислення першої похідної
        peaks_info, second_derivative = find_peaks_in_spectrum(smoothed_spectrum)
        plot_spectrum(spectrum, peaks_info, smoothed_spectrum)
        peak_indices = [idx for idx, _, _, _, _, _ in peaks_info]
        plot_second_derivative(second_derivative, peak_indices, first_derivative)
        plot_only_second_derivative(second_derivative, peak_indices)
        display_peak_info(peaks_info)
        save_peaks_info_to_file(peaks_info)

# Функція для відображення інформації про знайдені піки
def display_peak_info(peaks_info):
    peak_info_text.delete(1.0, tk.END)
    if peaks_info:
        peak_info_text.insert(tk.END, "Знайдені піки:\n")

        for i, (idx, amplitude, area, sigma_gaussian, fwhm, sigma_mnk) in enumerate(peaks_info, start=1):
            peak_info_text.insert(tk.END, f"{i}. Канал: {idx}, Амплітуда: {amplitude:.2f}, Площа: {area:.2f}, σ_гаус: {sigma_gaussian:.2f}, ΔE: {fwhm:.2f}, σ_мнк: {sigma_mnk:.2f}\n")

        peak_info_text.insert(tk.END, "\nЗначення сігми методом найменших квадратів (σ_мнк) відображає точність апроксимації кожного піка.")
    else:
        peak_info_text.insert(tk.END, "Піки не знайдено.")

# Графічний інтерфейс
root = tk.Tk()
root.title("Аналіз спектру та пошук піків")

# Кнопка для запуску аналізу
tk.Button(root, text="Аналізувати спектр", command=analyze_spectrum).grid(row=0, columnspan=3, pady=10)

# Вікно для виведення інформації про піки
peak_info_text = scrolledtext.ScrolledText(root, width=60, height=15)
peak_info_text.grid(row=1, columnspan=3, pady=10)

root.mainloop()
