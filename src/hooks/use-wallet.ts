import { useState, useCallback, useEffect } from "react";

declare global {
  interface Window {
    ethereum?: {
      isMetaMask?: boolean;
      request: (args: { method: string; params?: unknown[] }) => Promise<unknown>;
      on: (event: string, callback: (...args: unknown[]) => void) => void;
      removeListener: (event: string, callback: (...args: unknown[]) => void) => void;
    };
  }
}

export interface WalletState {
  address: string | null;
  isConnecting: boolean;
  isConnected: boolean;
  error: string | null;
  chainId: string | null;
}

export function useWallet() {
  const [state, setState] = useState<WalletState>({
    address: null,
    isConnecting: false,
    isConnected: false,
    error: null,
    chainId: null,
  });

  // Check if wallet is available
  const isWalletAvailable = typeof window !== "undefined" && !!window.ethereum;

  // Get short address for display
  const shortAddress = state.address
    ? `${state.address.slice(0, 6)}...${state.address.slice(-4)}`
    : null;

  // Connect wallet
  const connect = useCallback(async () => {
    if (!window.ethereum) {
      setState((prev) => ({
        ...prev,
        error: "请安装 MetaMask 或其他以太坊钱包",
      }));
      return null;
    }

    setState((prev) => ({ ...prev, isConnecting: true, error: null }));

    try {
      // Request account access
      const accounts = (await window.ethereum.request({
        method: "eth_requestAccounts",
      })) as string[];

      // Get chain ID
      const chainId = (await window.ethereum.request({
        method: "eth_chainId",
      })) as string;

      if (accounts.length > 0) {
        const address = accounts[0];
        setState({
          address,
          isConnecting: false,
          isConnected: true,
          error: null,
          chainId,
        });
        
        // Store in localStorage for persistence
        localStorage.setItem("wallet_address", address);
        
        return address;
      }
      
      return null;
    } catch (error) {
      const err = error as { code?: number; message?: string };
      let errorMessage = "连接钱包失败";
      
      if (err.code === 4001) {
        errorMessage = "用户拒绝了连接请求";
      } else if (err.message) {
        errorMessage = err.message;
      }

      setState((prev) => ({
        ...prev,
        isConnecting: false,
        error: errorMessage,
      }));
      
      return null;
    }
  }, []);

  // Disconnect wallet
  const disconnect = useCallback(() => {
    setState({
      address: null,
      isConnecting: false,
      isConnected: false,
      error: null,
      chainId: null,
    });
    localStorage.removeItem("wallet_address");
  }, []);

  // Sign message for authentication
  const signMessage = useCallback(async (message: string): Promise<string | null> => {
    if (!window.ethereum || !state.address) {
      setState((prev) => ({
        ...prev,
        error: "钱包未连接",
      }));
      return null;
    }

    try {
      const signature = (await window.ethereum.request({
        method: "personal_sign",
        params: [message, state.address],
      })) as string;

      return signature;
    } catch (error) {
      const err = error as { code?: number; message?: string };
      let errorMessage = "签名失败";
      
      if (err.code === 4001) {
        errorMessage = "用户拒绝了签名请求";
      } else if (err.message) {
        errorMessage = err.message;
      }

      setState((prev) => ({
        ...prev,
        error: errorMessage,
      }));
      
      return null;
    }
  }, [state.address]);

  // Listen for account changes
  useEffect(() => {
    if (!window.ethereum) return;

    const handleAccountsChanged = (accounts: unknown) => {
      const accs = accounts as string[];
      if (accs.length === 0) {
        disconnect();
      } else if (accs[0] !== state.address) {
        setState((prev) => ({
          ...prev,
          address: accs[0],
          isConnected: true,
        }));
        localStorage.setItem("wallet_address", accs[0]);
      }
    };

    const handleChainChanged = (chainId: unknown) => {
      setState((prev) => ({
        ...prev,
        chainId: chainId as string,
      }));
    };

    window.ethereum.on("accountsChanged", handleAccountsChanged);
    window.ethereum.on("chainChanged", handleChainChanged);

    return () => {
      window.ethereum?.removeListener("accountsChanged", handleAccountsChanged);
      window.ethereum?.removeListener("chainChanged", handleChainChanged);
    };
  }, [state.address, disconnect]);

  // Check for existing connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      if (!window.ethereum) return;

      const savedAddress = localStorage.getItem("wallet_address");
      if (!savedAddress) return;

      try {
        const accounts = (await window.ethereum.request({
          method: "eth_accounts",
        })) as string[];

        if (accounts.length > 0 && accounts[0].toLowerCase() === savedAddress.toLowerCase()) {
          const chainId = (await window.ethereum.request({
            method: "eth_chainId",
          })) as string;

          setState({
            address: accounts[0],
            isConnecting: false,
            isConnected: true,
            error: null,
            chainId,
          });
        } else {
          localStorage.removeItem("wallet_address");
        }
      } catch {
        localStorage.removeItem("wallet_address");
      }
    };

    checkConnection();
  }, []);

  return {
    ...state,
    shortAddress,
    isWalletAvailable,
    connect,
    disconnect,
    signMessage,
  };
}
