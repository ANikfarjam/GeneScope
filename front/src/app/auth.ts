import { createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, UserCredential, updateProfile } from "firebase/auth";
import { auth } from "./firebase";

// Type definition for authentication functions
export interface AuthResponse {
  user: UserCredential["user"] | null;
  error: string | null;
}


// Sign Up Function with Username
export const signUp = async (email: string, password: string, username: string): Promise<AuthResponse> => {
  try {
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);

    // Update Firebase user profile to store the username
    if (userCredential.user) {
      await updateProfile(userCredential.user, {
        displayName: username, // Stores the username in Firebase
      });
    }

    return { user: userCredential.user, error: null };
  } catch (error) {
    return { user: null, error: (error as Error).message };
  }
};


// Login Function
export const login = async (email: string, password: string): Promise<AuthResponse> => {
  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    return { user: userCredential.user, error: null };
  } catch (error) {
    return { user: null, error: (error as Error).message };
  }
};

// Logout Function
export const logout = async (): Promise<void> => {
  try {
    await signOut(auth);
  } catch (error) {
    console.error("Logout Error:", error);
  }
};
