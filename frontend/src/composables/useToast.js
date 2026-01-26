let toastInstance = null

export function setToastInstance(instance) {
  toastInstance = instance
}

export function useToast() {
  return {
    success: (message, description) => toastInstance?.success(message, description),
    error: (message, description) => toastInstance?.error(message, description),
    warning: (message, description) => toastInstance?.warning(message, description),
    info: (message, description) => toastInstance?.info(message, description)
  }
}
