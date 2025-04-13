import { UserCredential } from "firebase/auth";

// Type definition for authentication functions
export interface AuthResponse {
  user: UserCredential["user"] | null;
  error: string | null;
}

// Sign Up Function with Username
export const signUp = async (
  email: string,
  password: string,
  username: string
): Promise<AuthResponse> => {
  try {
    const { createUserWithEmailAndPassword, updateProfile } = await import(
      "firebase/auth"
    );
    const { auth } = await import("./firebase.client");

    const userCredential = await createUserWithEmailAndPassword(
      auth,
      email,
      password
    );

    if (userCredential.user) {
      await updateProfile(userCredential.user, {
        displayName: username,
      });
    }

    return { user: userCredential.user, error: null };
  } catch (error) {
    return { user: null, error: (error as Error).message };
  }
};

// Login Function
export const login = async (
  email: string,
  password: string
): Promise<AuthResponse> => {
  try {
    const { signInWithEmailAndPassword } = await import("firebase/auth");
    const { auth } = await import("./firebase.client");

    const userCredential = await signInWithEmailAndPassword(
      auth,
      email,
      password
    );
    return { user: userCredential.user, error: null };
  } catch (error) {
    return { user: null, error: (error as Error).message };
  }
};

// Logout Function
export const logout = async (): Promise<void> => {
  try {
    const { signOut } = await import("firebase/auth");
    const { auth } = await import("./firebase.client");

    await signOut(auth);
  } catch (error) {
    console.error("Logout Error:", error);
  }
};
