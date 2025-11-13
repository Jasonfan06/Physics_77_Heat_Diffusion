t_final = max_iter * dt
U_exact = (sigma0**2)/(sigma0**2 + 4*alpha*t_final) * \
          np.exp(-((X-0.5)**2 + (Y-0.5)**2)/(sigma0**2 + 4*alpha*t_final))

error_L2 = np.sqrt(np.mean((u - U_exact)**2))
print(f"L2 error = {error_L2:.6e}")
